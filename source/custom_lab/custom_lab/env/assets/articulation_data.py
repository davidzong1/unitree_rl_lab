from isaaclab.assets import ArticulationData
from isaaclab.utils import configclass
import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
try:
    from isaaclab.utils.math import quat_apply
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply


class ArticulationDataWithCM(ArticulationData):
    """Articulation data class that includes centroidal momentum information."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root_link_acc_w = TimestampedBuffer()

    @property
    def root_com_acc_w(self) -> torch.Tensor:
        """Root link acceleration in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular accelerations of the articulation root link's center of mass frame
        relative to the world.
        """
        return self.body_acc_w[:, 0]

    @property
    def root_link_acc_w(self) -> torch.Tensor:
        """Root link acceleration in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular accelerations of the root rigid body's actor frame relative to the
        world. Note that this is not the same as the root center of mass acceleration.
        """
        if self._root_link_acc_w.timestamp < self._sim_timestamp:
            root_link_ang_acc_w = self.root_com_acc_w[:, 3:6].clone()  # angular acceleration is the same
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_root_velocities().clone()

            # link_lin_acc_w = com_lin_acc_w + link_ang_acc_w x (r_o - r_c) + link_ang_vel_w x (link_ang_vel_w x (r_o - r_c))
            r_co = math_utils.quat_apply(pose[:, 3:7], -self.com_pos_b[:, 0, :])
            root_link_lin_acc_w = (
                self.root_com_acc_w[:, :3].clone()
                + torch.linalg.cross(root_link_ang_acc_w, r_co, dim=-1)
                + torch.linalg.cross(
                    velocity[:, 3:],
                    torch.linalg.cross(velocity[:, 3:], r_co, dim=-1),
                    dim=-1,
                )
            )
            self._root_link_acc_w.data = torch.cat((root_link_lin_acc_w, root_link_ang_acc_w), dim=-1)
            self._root_link_acc_w.timestamp = self._sim_timestamp
        return self._root_link_acc_w.data

    @property
    def root_link_lin_acc_w(self) -> torch.Tensor:
        """Root link linear acceleration in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear acceleration of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_acc_w[:, :3]

    @property
    def root_link_ang_acc_w(self) -> torch.Tensor:
        """Root link angular acceleration in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular acceleration of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_acc_w[:, 3:]

    @property
    def root_link_lin_acc_b(self) -> torch.Tensor:
        """Root link linear acceleration in base frame. Shape is (num_instances, 3).

        This quantity is the linear acceleration of the root rigid body's actor frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_acc_w)

    @property
    def root_link_ang_acc_b(self) -> torch.Tensor:
        """Root link angular acceleration in base world frame. Shape is (num_instances, 3).

        This quantity is the angular acceleration of the root rigid body's actor frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_acc_w)
