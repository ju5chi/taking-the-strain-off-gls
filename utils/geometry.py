from .constants import PREC


def compute_area_of_triangle(p1: tuple[float, float], 
                             p2: tuple[float, float],
                             p3: tuple[float, float]
                             ) -> float:
    '''
        computes the area of the triangle with vertices p1, p2, p3

        Source:
        https://en.wikipedia.org/wiki/Shoelace_formula#The_polygon_area_formulas
    '''
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return 0.5 * abs(x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3)

def compute_area_of_trapezium(p1: tuple[float, float],
                              p2: tuple[float, float], 
                              p3: tuple[float, float], 
                              p4: tuple[float, float]
                              ) -> float:
    '''
        computes the area of the trapezium with vertices p1, p2, p3, p4 (these
        have to be ordered in (counter-) clockwise direction)

        Source:
        https://en.wikipedia.org/wiki/Shoelace_formula#The_polygon_area_formulas
    '''
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    return 0.5 * abs((y1 + y2)*(x1 - x2) + (y2 + y3)*(x2 - x3) + (y3 + y4)*(x3 - x4) + (y4 + y1)*(x4 - x1))

def find_intersection_of_line_segments(ls1: tuple[tuple[float, float], tuple[float, float]], 
                                       ls2: tuple[tuple[float, float], tuple[float, float]]
                                       ) -> tuple[float, float]:
    '''
        computes the intersection of two line segments; it is assumed that such an
        intersection exists, otherwise an error will be thrown

        Source:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
       ----
        ls1:
            the coordinates of the first line segment
        ls2:
            the coordinates of the second line segment
       ----
        Returns:
            the intersection as a tuple
    '''

    (x1, y1), (x2, y2) = ls1
    (x3, y3), (x4, y4) = ls2

    # compute determinant to check for parallel lines
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # the following has to be true by assumption that the line segments are not parallel
    if abs(denom) < PREC:
        raise RuntimeError('The line segments are parallel.')

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    # check if the solution is actually an intersection of the segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    else:
        raise RuntimeError('The solution found is not an intersection.')

def linear_interpolation_at(x: float, 
                            p1: tuple[float, float], 
                            p2: tuple[float, float]
                            ) -> tuple[float, float]:
    '''
        finds y s.t. (x, y) is a point on the line segment between p1 and p2; it
        is assumed that the first coordinates of p1 and p2 differ
       ---
        x:
            the first coordinate value at which the linear interpolation should 
            be evaluated
        p1:
            a tuple containing the coordinates of an endpoint of a line segment
        p2: 
            see p1
       ----
        Returns:
            a tuple (x, y), where y is such that (x, y) lies on the line segment
            between p1 and p2
    '''
    denom = p1[0] - p2[0]

    if abs(denom) < PREC:
        raise RuntimeError('The first coordinates of p1 and p2 coincide.')

    alpha = (x - p2[0])/denom

    return (x, alpha*p1[1] + (1-alpha)*p2[1])

def area_between_warping_curves(time: list, t: list, t_hat: list):
    '''
        computes the area between two non-decreasing curves, i.e. when a curve
        is given by linear interpolation of points (x_1, y_1), ..., (x_n, y_n), then
        it is said to be non-decreasing if x_i <= x_{i+1} and y_i <= y_{i+1} for
        all i.
       ----
        time:
            a list containing all time points
        t:
            a list containing the tuples of floats that define the curve
        t_hat:
            see t
       ----
        Returns:
            the area between t and t_hat 
    '''
    
    # first phase: compute all pairs of (a, b) and (a_hat, b_hat)
    A, B, A_hat, B_hat = [], [], [], []

    for s, (start_points, end_points) in zip([t, t_hat], [(A, B), (A_hat, B_hat)]):
        s_x_coord_fixed = False
        s_old_x, s_old_y = None, None
        time_index = 0
        cache = []

        for i in range(len(s)):
            s_i_x, s_i_y = s[i]
            s_next_x, s_next_y = None, None

            # not reached last tuple in s yet
            if i < len(s)-1:
                s_next_x, s_next_y = s[i+1]
                completed_missing_values = False

                # increment the time index if there is a difference between the current and the next time
                if s_next_x - s_i_x >= PREC:
                    time_index += 1

                # check if there are times missing in the entries of s
                while s_next_x - time[time_index] >= PREC:
                    # append the current points first, before doing interpolation
                    # if this loop is entered on the first iteration
                    if i <= 0:
                        start_points.append((s_i_x, s_i_y))
                        end_points.append((s_i_x, s_i_y))

                    time_now = time[time_index]
                    # interpolate linearly between s_i and s_next
                    _, y = linear_interpolation_at(time_now, (s_i_x, s_i_y), (s_next_x, s_next_y))
                    # append to the start/end points
                    start_points.append((time_now, y))
                    end_points.append((time_now, y))

                    completed_missing_values = True
                    time_index += 1

                # if the while loop above has been entered, then continue with the next iteration
                # in the for loop
                if completed_missing_values:
                    continue

                # temporarily save the values
                cache.append((s_i_x, s_i_y))

                # if a streak is about to end, consider the cache and save the right values
                if abs(s_next_x - s_i_x) >= PREC:
                    x_value = cache[0][0]
                    a = max(cache, key = lambda tup: tup[1])[1]
                    b = min(cache, key = lambda tup: tup[1])[1]

                    start_points.append((x_value, a))
                    end_points.append((x_value, b))

                    cache = []

            # last tuple is reached
            else:
                if len(cache) > 0:
                    cache.append((s_i_x, s_i_y))
                    x_value = cache[0][0]
                    a = max(cache, key = lambda tup: tup[1])[1]
                    b = min(cache, key = lambda tup: tup[1])[1]

                    start_points.append((x_value, a))
                    end_points.append((x_value, b))
                
                else:
                    start_points.append((s_i_x, s_i_y))
                    end_points.append((s_i_x, s_i_y))
    
    # second phase: 
    area = 0
    n = len(A)-1

    assert len(A) == len(A_hat), f'len(A) = {len(A)}; len(A_hat) = {len(A_hat)}'

    for i in range(n):
        area_domain = 0

        (t_i, a), (t_next, b) = A[i], B[i+1]
        (t_hat_i, a_hat), (t_hat_next, b_hat) = A_hat[i], B_hat[i+1]

        # case 1: a = a_hat
        if abs(a - a_hat) < PREC:
            if abs(b - b_hat) >= PREC:
                area_domain = compute_area_of_triangle((t_i, a), (t_next, b), (t_next, b_hat))
        
        else:
            # b = b_hat
            if abs(b - b_hat) < PREC:
                area_domain = compute_area_of_triangle((t_i, a), (t_i, a_hat), (t_next, b_hat))
            
            # there is no intersection of the line segments a<->b and a_hat<->b_hat
            elif (a < a_hat and b < b_hat) or (a > a_hat and b > b_hat):
                area_domain = compute_area_of_trapezium((t_i, a), (t_next, b), (t_next, b_hat), (t_i, a_hat))

            # there is an intersection
            elif (a < a_hat and b > b_hat) or (a > a_hat and b < b_hat):
                ab = ((t_i, a), (t_next, b))
                ab_hat = ((t_i, a_hat), (t_next, b_hat))
                intersection_point = find_intersection_of_line_segments(ab, ab_hat)

                area_domain = compute_area_of_triangle((t_i, a), (t_i, a_hat), intersection_point)\
                    + compute_area_of_triangle((t_next, b), (t_next, b_hat), intersection_point)

        area += area_domain
    
    return area


if __name__ == '__main__':
    # tests:
    time = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = [(0.0, 0.0), (0.25, 0.5), (0.5, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.0)]
    t_hat = [(tup[1], tup[0]) for tup in t]
    diag = [(ti, ti) for ti in time]

    print(area_between_warping_curves(time, t, t_hat))
    print(area_between_warping_curves(time, t, diag))
    print(area_between_warping_curves(time, t_hat, diag))