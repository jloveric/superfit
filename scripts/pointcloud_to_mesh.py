"""
Convert a point cloud PLY (or other Open3D-supported format) into a triangle mesh
suitable for SuperFit's mesh_to_assembly pipeline.

Uses Open3D for Poisson / ball-pivoting reconstruction and, when needed, a voxel
occupancy + marching-cubes fallback (scikit-image) that produces closed meshes.
"""
import argparse
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from skimage import measure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a triangle mesh from a point cloud using Open3D."
    )
    parser.add_argument(
        "--preset",
        choices=("default", "dense"),
        default="default",
        help="Quality preset. dense raises detail-focused reconstruction settings.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input point cloud path (e.g. .ply).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output mesh path (e.g. .obj or .ply).",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "poisson", "bpa", "voxel"),
        default="auto",
        help="Reconstruction method. auto tries Poisson, then voxel if not watertight.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=500_000,
        help="Downsample to roughly this many points before reconstruction.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for downsampling. Auto-estimated from --target-points when omitted.",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=10,
        help="Octree depth for Poisson reconstruction.",
    )
    parser.add_argument(
        "--poisson-scale",
        type=float,
        default=1.2,
        help="Grid scale for Poisson reconstruction.",
    )
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.01,
        help="Remove Poisson vertices with density below this quantile.",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=None,
        help="Radius for normal estimation. Defaults to 2 * voxel size.",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=30,
        help="Max neighbors for normal estimation.",
    )
    parser.add_argument(
        "--normal-orientation",
        choices=("consistent", "camera"),
        default="consistent",
        help="How to orient estimated normals before reconstruction.",
    )
    parser.add_argument(
        "--normal-consistent-k",
        type=int,
        default=40,
        help="Neighbors used for consistent tangent-plane normal orientation.",
    )
    parser.add_argument(
        "--remove-statistical-outliers",
        action="store_true",
        help="Apply statistical outlier removal before reconstruction.",
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=192,
        help="Grid resolution for voxel reconstruction.",
    )
    parser.add_argument(
        "--voxel-dilation",
        type=int,
        default=4,
        help="Morphological dilation iterations for voxel reconstruction.",
    )
    parser.add_argument(
        "--voxel-closing",
        type=int,
        default=3,
        help="Morphological closing iterations for voxel reconstruction.",
    )
    parser.add_argument(
        "--require-watertight",
        action="store_true",
        help="Exit with an error if the output mesh is not watertight.",
    )
    return parser.parse_args()


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"No points found in {path}")
    if not pcd.has_points():
        raise ValueError(f"Input is not a point cloud: {path}")
    return pcd


def normalize_point_cloud(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    points = np.asarray(pcd.points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 0.0:
        raise ValueError("Point cloud has zero extent.")
    normalized = centered / scale
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(normalized)
    if pcd.has_colors():
        out.colors = pcd.colors
    if pcd.has_normals():
        out.normals = pcd.normals
    return out, center, scale


def denormalize_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    vertices = mesh.vertices * scale + center
    return trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
        process=False,
        metadata=mesh.metadata,
    )


def estimate_voxel_size(pcd: o3d.geometry.PointCloud, target_points: int) -> float:
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    extent = np.maximum(extent, 1e-6)
    volume = float(np.prod(extent))
    voxel = (volume / target_points) ** (1.0 / 3.0)
    for _ in range(12):
        down = pcd.voxel_down_sample(voxel)
        n_down = len(down.points)
        if n_down > target_points * 1.25:
            voxel *= 1.15
        elif n_down < target_points * 0.75:
            voxel *= 0.85
        else:
            break
    return float(voxel)


def downsample_point_cloud(
    pcd: o3d.geometry.PointCloud,
    target_points: int,
    voxel_size: float | None,
) -> tuple[o3d.geometry.PointCloud, float]:
    n_points = len(pcd.points)
    if n_points <= target_points and voxel_size is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        voxel_size = float(np.max(extent) / 100.0)
        return pcd, voxel_size

    if voxel_size is None:
        voxel_size = estimate_voxel_size(pcd, target_points)

    down = pcd.voxel_down_sample(voxel_size)
    if len(down.points) > target_points * 2:
        ratio = target_points / len(down.points)
        down = down.random_down_sample(ratio)
    return down, voxel_size


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
    orientation: str,
    consistent_k: int,
) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if orientation == "consistent":
        pcd.orient_normals_consistent_tangent_plane(consistent_k)
    else:
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([2.0, 2.0, 2.0], dtype=np.float64)
        )
    pcd.normalize_normals()


def apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset != "dense":
        return

    args.target_points = max(args.target_points, 1_000_000)
    args.poisson_depth = max(args.poisson_depth, 12)
    args.normal_max_nn = max(args.normal_max_nn, 60)
    args.normal_consistent_k = max(args.normal_consistent_k, 80)
    args.voxel_resolution = max(args.voxel_resolution, 256)
    args.density_quantile = min(args.density_quantile, 0.008)


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    scale: float,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=True,
    )
    if density_quantile > 0.0:
        densities = np.asarray(densities)
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def reconstruct_bpa(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.TriangleMesh:
    radii = o3d.utility.DoubleVector(
        [voxel_size, voxel_size * 2.0, voxel_size * 4.0]
    )
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def reconstruct_voxel(
    points: np.ndarray,
    resolution: int,
    dilation: int,
    closing: int,
) -> trimesh.Trimesh:
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    idx = ((points + 1.0) / 2.0 * (resolution - 1)).astype(np.int32)
    idx = np.clip(idx, 0, resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    grid = binary_dilation(grid, iterations=dilation)
    grid = binary_closing(grid, iterations=closing)
    grid = binary_fill_holes(grid)
    verts, faces, _, _ = measure.marching_cubes(grid.astype(np.float32), level=0.5)
    verts = verts / (resolution - 1) * 2.0 - 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def open3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        raise ValueError("Reconstruction produced an empty mesh.")
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    trimesh.repair.fill_holes(tm)
    tm.fix_normals()
    tm.merge_vertices()
    tm.update_faces(tm.nondegenerate_faces())
    tm.update_faces(tm.unique_faces())
    return tm


def print_mesh_stats(mesh: trimesh.Trimesh) -> None:
    print(f"  vertices: {len(mesh.vertices):,}")
    print(f"  faces: {len(mesh.faces):,}")
    print(f"  watertight: {mesh.is_watertight}")
    print(f"  winding consistent: {mesh.is_winding_consistent}")


def main() -> None:
    args = parse_args()
    apply_preset_defaults(args)

    print(f"Loading point cloud: {args.input}")
    pcd = load_point_cloud(args.input)
    print(f"  points: {len(pcd.points):,}")

    if args.remove_statistical_outliers:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
        print(f"  after outlier removal: {len(pcd.points):,}")

    pcd, center, scale = normalize_point_cloud(pcd)
    pcd, voxel_size = downsample_point_cloud(
        pcd,
        target_points=args.target_points,
        voxel_size=args.voxel_size,
    )
    print(f"  downsampled points: {len(pcd.points):,} (voxel size {voxel_size:.6f})")

    method = args.method
    mesh = None

    if method in ("auto", "poisson", "bpa"):
        normal_radius = args.normal_radius
        if normal_radius is None:
            normal_radius = voxel_size * 2.0
        estimate_normals(
            pcd,
            radius=normal_radius,
            max_nn=args.normal_max_nn,
            orientation=args.normal_orientation,
            consistent_k=args.normal_consistent_k,
        )

        if method in ("auto", "poisson"):
            print("Reconstructing mesh with Poisson...")
            o3d_mesh = reconstruct_poisson(
                pcd,
                depth=args.poisson_depth,
                scale=args.poisson_scale,
                density_quantile=args.density_quantile,
            )
            mesh = open3d_to_trimesh(o3d_mesh)
        else:
            print("Reconstructing mesh with ball pivoting...")
            o3d_mesh = reconstruct_bpa(pcd, voxel_size=voxel_size)
            mesh = open3d_to_trimesh(o3d_mesh)

        if method == "auto" and not mesh.is_watertight:
            print("Poisson mesh is not watertight; retrying Poisson with denser settings...")
            retry_depth = min(args.poisson_depth + 1, 13)
            retry_quantile = args.density_quantile * 0.5
            retry_mesh = reconstruct_poisson(
                pcd,
                depth=retry_depth,
                scale=args.poisson_scale,
                density_quantile=retry_quantile,
            )
            mesh = open3d_to_trimesh(retry_mesh)
            if not mesh.is_watertight:
                print("Retry still not watertight; falling back to voxel reconstruction...")
                method = "voxel"

    if method == "voxel":
        print("Reconstructing mesh with voxel occupancy + marching cubes...")
        points = np.asarray(pcd.points, dtype=np.float64)
        mesh = reconstruct_voxel(
            points,
            resolution=args.voxel_resolution,
            dilation=args.voxel_dilation,
            closing=args.voxel_closing,
        )

    mesh = denormalize_mesh(mesh, center=center, scale=scale)
    mesh.export(args.output)

    print(f"Saved mesh: {args.output}")
    print_mesh_stats(mesh)

    if args.require_watertight and not mesh.is_watertight:
        print(
            "Mesh is not watertight. Try --method voxel, increase "
            "--voxel-dilation, or raise --target-points.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
