import torch
import numpy as np
import os


class PinLabConvert:
    def __init__(self, robot_name: str, pinJointName: list[str], labJointName: list[str]) -> None:
        if pinJointName.__len__() != labJointName.__len__():
            print("Pin joint is: ", pinJointName)
            print("Lab joint is: ", labJointName)
            raise ValueError("The length of pinJointName and labJointName must be the same.")
        self.q_to_lab = [0, 1, 2, 3, 4, 5, 6]
        self.q_to_pin = [0, 1, 2, 3, 4, 5, 6]
        self.v_to_lab = [0, 1, 2, 3, 4, 5]
        self.v_to_pin = [0, 1, 2, 3, 4, 5]
        self.joint_to_lab = []
        self.joint_to_pin = []
        to_lab_joint_sort = []
        to_pin_joint_sort = []
        for i in range(pinJointName.__len__()):
            if pinJointName[i] in labJointName:
                to_lab_joint_sort.append(labJointName.index(pinJointName[i]))
            else:
                print("Pin joint is: ", pinJointName)
                print("Lab joint is: ", labJointName)
                raise ValueError(f"Pin joint {pinJointName[i]} not found in lab joint names.")
        for i in range(labJointName.__len__()):
            if labJointName[i] in pinJointName:
                to_pin_joint_sort.append(pinJointName.index(labJointName[i]))
            else:
                print("Pin joint is: ", pinJointName)
                print("Lab joint is: ", labJointName)
                raise ValueError(f"Lab joint {labJointName[i]} not found in pin joint names.")
        self.q_to_lab.extend([x + 7 for x in to_lab_joint_sort])
        self.q_to_pin.extend([x + 7 for x in to_pin_joint_sort])
        self.v_to_lab.extend([x + 6 for x in to_lab_joint_sort])
        self.v_to_pin.extend([x + 6 for x in to_pin_joint_sort])
        self.joint_to_lab = to_lab_joint_sort
        self.joint_to_pin = to_pin_joint_sort
        dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
        pin2lab_dir = os.path.join(dir_path, "pin2lab")
        robot_dir_path = os.path.join(pin2lab_dir, robot_name + "_pin_lab_map")
        if not os.path.isdir(robot_dir_path):
            try:
                os.makedirs(robot_dir_path, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create directory {robot_dir_path}: {e}")
        self.save_mappings(os.path.join(robot_dir_path, "pin_lab_mapping.csv"))

    def save_mappings(self, filepath: str) -> None:
        """
        保存映射到 CSV 文件。CSV 列顺序：
        q_to_lab, q_to_pin, v_to_lab, v_to_pin, joint_to_lab, joint_to_pin
        """
        import csv

        lists = [
            self.q_to_lab,
            self.q_to_pin,
            self.v_to_lab,
            self.v_to_pin,
            self.joint_to_lab,
            self.joint_to_pin,
        ]
        headers = [
            "q_to_lab",
            "q_to_pin",
            "v_to_lab",
            "v_to_pin",
            "joint_to_lab",
            "joint_to_pin",
        ]
        max_len = max(len(l) for l in lists)
        with open(filepath, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(max_len):
                row = [(str(l[i]) if i < len(l) else "") for l in lists]
                writer.writerow(row)

    def q_lab_to_pin(
        self,
        gen_coord: torch.Tensor,
    ) -> torch.Tensor:
        return gen_coord[:, self.q_to_pin]

    def q_pin_to_lab(
        self,
        gen_coord: torch.Tensor,
    ) -> torch.Tensor:
        return gen_coord[:, self.q_to_lab]

    def v_lab_to_pin(
        self,
        gen_vel: torch.Tensor,
    ) -> torch.Tensor:
        return gen_vel[:, self.v_to_pin]

    def v_pin_to_lab(
        self,
        gen_vel: torch.Tensor,
    ) -> torch.Tensor:
        return gen_vel[:, self.v_to_lab]

    def f_lab_to_pin(self, gen_force: torch.Tensor) -> torch.Tensor:
        return gen_force[:, self.v_to_pin]

    def f_pin_to_lab(self, gen_force: torch.Tensor) -> torch.Tensor:
        return gen_force[:, self.v_to_lab]

    def m_lab_to_pin(self, mass_matrix: torch.Tensor) -> torch.Tensor:
        return mass_matrix[:, self.v_to_pin][:, self.v_to_pin]

    def m_pin_to_lab(self, mass_matrix: torch.Tensor) -> torch.Tensor:
        return mass_matrix[:, self.v_to_lab][:, self.v_to_lab]

    def joint_lab_to_pin(self, joint_order: torch.Tensor) -> torch.Tensor:
        return joint_order[:, self.joint_to_pin]

    def joint_pin_to_lab(self, joint_order: torch.Tensor) -> torch.Tensor:
        return joint_order[:, self.joint_to_lab]
