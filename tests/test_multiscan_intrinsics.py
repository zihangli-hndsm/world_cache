import numpy as np

from audit_multiscan_rgbd_projection import scaled_intrinsic_parameters


def test_column_major_rgb_intrinsics_scale_center_and_focal_lengths():
    # Column-major [fx, 0, 0, 0, fy, 0, cx, cy, 1].
    values = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 450.0, 300.0, 1.0]
    fx, fy, cx, cy = scaled_intrinsic_parameters(values, 0.25, 0.5)

    np.testing.assert_allclose([fx, fy, cx, cy], [225.0, 400.0, 112.5, 150.0])
    # The principal-point pixel must generate the optical-axis ray.
    np.testing.assert_allclose([(cx - cx) / fx, (cy - cy) / fy, 1.0], [0.0, 0.0, 1.0])
