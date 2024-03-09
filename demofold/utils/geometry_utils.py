import torch


def dot(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """v1, v2: [*, 3]"""
    return torch.sum(v1 * v2, dim=-1)


def norm(v: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Compute Norm of Vec3Array, clipped to epsilon."""
    # To avoid NaN on the backward pass, we must use maximum before the sqrt
    norm2 = dot(v, v)
    if epsilon:
        norm2 = torch.clamp(norm2, min=epsilon**2)
    return torch.sqrt(norm2)


# def cross(a: torch.Tensor, b: torch.Tensor):
#     """Compute cross product between 'a' and 'b'."""
#     """a,b:[*, 3]"""
#     new_x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
#     new_y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
#     new_z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
#     return torch.stack((new_x, new_y, new_z), dim=-1)


def calc_dihedral(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    d: torch.Tensor,
    degree: bool = False
) -> torch.Tensor:
    """
    a,b,c,d: [*, 3]
    return: [*]
    """
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = v1.cross(v2, dim=-1)
    c2 = v3.cross(v2, dim=-1)
    c3 = c2.cross(c1, dim=-1)

    v2_mag = norm(v2)
    result = torch.atan2(dot(c3, v2), v2_mag * dot(c1, c2))
    if degree:
        result = torch.rad2deg(result)
        
    return result
