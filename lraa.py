import numpy as np
from collections import OrderedDict

A = np.array([
    [0, -1],
    [0, 1],
    [0, 1],
    [-1, 0],
    [1, 1],
])

b = np.array([
    0,
    1,
    5,
    0,
    1
])

def vertex_enumeration_2d(A, b, eps=1e-6):
    # Assume Ax <= b
    # Assume the system is feasible and bounded
    # Return an OrderedDict with keys=vertex coordinates, values=[active constraints, in increasing radian order (counter-clockwise)]

    rads = np.arctan2(A[:,1], A[:,0])
    rads[rads<0] = 2*np.pi + rads[rads<0]  # Convert negative radians to positive
    
    # Now, rads start at 0 pointing towards positive x-axis, and goes counter-clockwise
    rads_order = np.argsort(rads)

    A_normalized = A / np.linalg.norm(A, axis=1).reshape(len(A), 1)

    V2active_constraints = OrderedDict()  # key is the vertex coordinates, value is the list of constraint indices (sorted in radian order)
    for order_i, i in enumerate(rads_order.tolist()):
        a_i = A[i]
        b_i = b[i]
        a_normalized_i = A_normalized[i]
        for j in np.hstack([rads_order[order_i+1:], rads_order[:order_i]]).tolist():
            a_j = A[j]
            b_j = b[j]
            a_normalized_j = A_normalized[j]

            if abs(a_normalized_i.dot(a_normalized_j)) >= 1-eps:
                # the two lines are parallel to each other
                continue

            A_ij = np.vstack([a_i, a_j])
            b_ij = np.hstack([b_i, b_j])

            # if the two lines are not parallel to each other, they are linearly independent and spans the entire 2D space
            # A_ij.T.dot(A_ij) will be invertible
            intersection_pt = np.linalg.inv(A_ij.T.dot(A_ij)).dot(A_ij.T.dot(b_ij))
            
            # Check if the intersection point is within the polygon
            if (A.dot(intersection_pt) <= b + eps).sum() < len(b):
                continue
            
            if len(V2active_constraints) == 0:
                V2active_constraints[tuple(intersection_pt.tolist())] = [i, j]
                break
            else:
                existing_pts = np.array(list(V2active_constraints.keys()))
                dist = np.linalg.norm(existing_pts - intersection_pt, axis=1)
                argmin_dist = np.argmin(dist)
                if dist[argmin_dist] <= eps:
                    # Ensure numerical instability does not generate duplicate points differing slightly
                    if j not in V2active_constraints[list(V2active_constraints.keys())[argmin_dist]]:
                        V2active_constraints[list(V2active_constraints.keys())[argmin_dist]].append(j)
                else:
                    V2active_constraints[tuple(intersection_pt.tolist())] = [i, j]
                    break
    
    return V2active_constraints

vertex_enumeration_2d(A, b)