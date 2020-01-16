from math import cos, pi, tanh
from functools import partial


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


def anneal_cospow(start, end, proportion):
    power = 5

    cos_val = 0.5 * (cos(pi * proportion) + 1) + 1
    cos_val = power ** cos_val - power
    cos_val = cos_val / (power ** 2 - power)

    return end + (start - end) * cos_val


def anneal_poly(start, end, proportion, power=0.9):
    return (start - end) * (1 - proportion) ** power + end


def anneal_tanh(start, end, proportion, lower=-6, upper=3):
    return end + (start - end) / 2 * (1 - tanh(lower + (upper - lower) * proportion))


def anneal_flat(start, end, proportion):
    return start


class Phase:
    def __init__(self, start, end, n_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


def phase(name, lr_min, lr_max, proportion, **kwargs):
    return name, lr_min, lr_max, proportion, kwargs


class PhaseScheduler:
    def __init__(
        self,
        optimizer,
        lr,
        n_iter,
        phases=[('linear', 1 / 25, 1, 0.25), ('cos', 1, 1 / 1e5, 0.75)],
    ):
        self.optimizer = optimizer

        phase_map = {
            'linear': anneal_linear,
            'cos': anneal_cos,
            'cospow': anneal_cospow,
            'poly': anneal_poly,
            'tanh': anneal_tanh,
            'flat': anneal_flat,
        }

        self.lr_phase = []

        for phase in phases:
            if len(phase) == 4:
                phase_name, lr_from, lr_to, prop = phase
                phase_fn = phase_map[phase_name]

            else:
                phase_name, lr_from, lr_to, prop, phase_args = phase
                phase_fn = partial(phase_map[phase_name], **phase_args)

            lr_min = lr_from * lr
            lr_max = lr_to * lr
            phase_iter = n_iter * prop

            phase_item = Phase(lr_min, lr_max, phase_iter, phase_fn)

            self.lr_phase.append(phase_item)

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr


def cycle_scheduler(
    optimizer,
    lr,
    n_iter,
    div=25,
    div_final=1e5,
    warmup=0.25,
    plateau=0,
    decay=('cos', 'cos'),
):
    phases = []

    if warmup > 0:
        phases.append((decay[0], 1 / div, 1, warmup))

    if plateau > 0:
        phases.append(('flat', 1, 1, plateau))

    phases.append((decay[1], 1, 1 / div_final, 1 - warmup - plateau))

    return PhaseScheduler(optimizer, lr, n_iter, phases)
