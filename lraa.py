import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colormaps

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

def representation_transform_HV(A, b):
    return list(vertex_enumeration_2d(A, b).keys())

def representation_transform_VH(V, eps=1e-6):
    A = []
    b = []
    for v1, v2 in zip(
        np.vstack([V[-1], V[:-1]]),  # such that the normal of the first line segment will be either horizontal (towards strict right) or pointing upwards
        V
    ):
        v12 = np.array(v2) - np.array(v1)

        if abs(v12[0]) <= eps:
            # v12 is a vertical line
            a = np.array([1, 0])
        elif abs(v12[1]) <= eps:
            # v12 is a horizontal line
            a = np.array([0, 1])
        else:
            a2 = 1  # With only the line segment, we have degree of freedom = 1. Fixing one coordinate 'consumes' that
            # Orthogonality: v12[0] * a1 + v12[1] * a2 = 0
            a1 = -v12[1] * a2 / v12[0]

            a = np.array([a1, a2])
            a = a / np.linalg.norm(a)

        b_i = a.dot(v1)

        if len(A) == 0:
            # The normal of the first line segment will be either horizontal (towards strict right) or pointing upwards
            if a[1] < 0:
                a *= -1
                b_i *= -1

        else:
            last_a = A[-1]
            
            # By convexity, cross product last_a × a must be pointing towards the reader
            if np.cross(
                np.hstack([last_a, 0]), 
                np.hstack([a, 0])
            )[2] < 0:
                a *= -1
                b_i *= -1

        A.append(a)
        b.append(b_i)
    
    return np.array(A).tolist(), np.array(b).tolist()

def standardize(A, b):
    V_representation = representation_transform_HV(A, b)
    return representation_transform_VH(V_representation)

def plot_polygon(A, b, constraints=False, linspace_num=100):
    V = representation_transform_HV(A, b)
    V = np.array(V)
    bbox = np.array([  # TODO: change this to dynamic
        [0, 1],
        [0, 1]
    ])

    x_range, y_range = np.ptp(bbox, axis=1)
    x = np.linspace(bbox[0][0] - x_range*0.1, 
                    bbox[0][1] + x_range*0.1, 
                    linspace_num)


    plt.figure(figsize=(3,3))
    plt.fill(V[:,0], V[:,1], '0.8')

    if constraints:
        cmap = colormaps['tab10']
        n = 0
        for a, b_i in zip(A, b):
            plot_kwargs = {
                'color': cmap.colors[n % 10],
                'label': f'{a[0]}x+{a[1]}y={b_i}'
            }
            if a[0] == 0:  # vertical line
                plt.axvline(x=b_i/a[1], **plot_kwargs)
            elif a[1] == 0:  # horizontal line
                plt.axhline(y=b_i/a[0], **plot_kwargs)
            else:
                constraint_y = (b_i - a[0] * x) / a[1]
                plt.plot(x, constraint_y, **plot_kwargs)
            n+=1
        plt.legend()
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(bbox[0][0]-x_range*0.1, bbox[0][1]+x_range*0.1)
    ax.set_ylim(bbox[0][0]-y_range*0.1, bbox[0][1]+y_range*0.1)
    plt.show()


print(standardize(A, b))

plot_polygon(A, b, constraints=True)