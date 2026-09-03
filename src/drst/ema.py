import copy
import torch


class EMATeacher:
    def __init__(self, student, decay=0.999):
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.decay = decay

        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        for teacher_p, student_p in zip(
            self.teacher.parameters(),
            self.student.parameters(),
        ):
            teacher_p.mul_(self.decay)
            teacher_p.add_(
                student_p,
                alpha=1.0 - self.decay,
            )

    def eval(self):
        self.teacher.eval()

    def model(self):
        return self.teacher
