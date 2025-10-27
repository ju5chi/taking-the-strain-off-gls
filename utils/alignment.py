import numpy as np
import itertools
import dtw
import fdasrsf as fs

from .constants import PREC
from .numerical_methods import differentiate, antiderivative

def align_fdasrsf(time: np.ndarray, 
                  curves: np.ndarray
                  ) -> dict:
    '''
        wrapper for fdasrsf alignment
       ----
        time:
            an array of time points that all curves share
        curves:
            an array containing the curves in along its second axis and their
            temporal evolution along its first axis
       ----
        Returns:
            output dictionary
    '''
    alignment = fs.fdawarp(curves, time)
    alignment.srsf_align(method='mean', parallel=True, verbose=False)

    output = {'time': time,
              'aligned_curves': alignment.fn,
              'time_warping': alignment.gam
              }

    return output

def dtw_barycentric_averaging(time: np.ndarray, 
                              curves: np.ndarray, 
                              step_pattern: dtw.StepPattern = dtw.symmetricP0, 
                              maxcount: int = 30, 
                              epsilon: float = 1e-10,
                              verbose: bool = False
                              ) -> np.ndarray:
    '''
        implements the DTW Barycentric Averaging (DBA) algorithm
       ----
        time:
            an array of time points that all curves share
        curves:
            an array containing the curves in along its second axis and their
            temporal evolution along its first axis
        step_pattern:
            the step pattern used by the dtw.dtw function
        maxcount:
            the maximum number of iterations of DBA
        epsilon:
            tolerance level to assess the convergence of DBA
        verbose:
            whether text output about the convergence should be generated
       ----
        Returns:
            the mean curve found by DBA
    '''

    n_curves = curves.shape[1]
    mean_curve = np.mean(curves, axis=1)
    temp_aligned_curves = []
    count = 0

    while count < maxcount:
        
        # compute the warping curves for aligning each of the curves to the 
        # current mean
        warped_curves = []
        for i in range(n_curves):
            alignment = dtw.dtw(curves[:, i], 
                                mean_curve, 
                                step_pattern=step_pattern
                                )
            
            time_temp = time[alignment.index2]
            warp_temp = curves[alignment.index1, i]
            warped_curves.append(np.stack([time_temp, warp_temp], axis=1))

        next_mean_curve = []

        for t in time:
            values_at_t = []
            for w_curve in warped_curves:
                time_mask = np.abs(w_curve[:, 0] - t) <= PREC
                values_at_t += list(w_curve[time_mask, 1])

            if len(values_at_t) <= 0:
                raise ValueError(f'Time {t:.3f} has no values.')
            
            next_mean_curve.append(np.mean(values_at_t))

        next_mean_curve = np.array(next_mean_curve)

        # breaking condition, when converged w.r.t. l_\infty norm
        if np.all(np.abs(next_mean_curve - mean_curve) <= epsilon):
            break

        mean_curve = next_mean_curve
        count += 1

    if verbose:
        if count < maxcount:
            print(f'DBA: converged in {count} iterations.')
        else:
            print(f'DBA: not converged; maxcount of {maxcount} reached.')

    return mean_curve

def align_dtw(time: np.ndarray, 
              curves: np.ndarray,
              step_pattern: dtw.StepPattern = dtw.symmetricP0, 
              alignment_type: str = 'combinations', 
              means_time_tup: None | tuple = None,
              dba_args: None | dict = None):
    '''
        wrapper for dtw alignment, that provides several options to align a group of curves
       ----
        time:
            an array of time points that all curves share
        curves:
            an array containing the curves in along its second axis and their
            temporal evolution along its first axis
        step_pattern:
            the step pattern used by the dtw.dtw function
        alignment_type: 
            specifies the type of alignment that should be used; possible options are
            "combinations" (default), "permutations", "mean", "derivative", "distance measure",
            "integration", "segmental mean"
        means_time_tup:
            segmental means and corresponding time grids
        dba_args:
            arguments for calling the function `dtw_barycentric_averaging` when
            alignment_type == "mean" 
       ----
        Returns:
            output dictionary
    '''

    n_segments = curves.shape[1]
    new_time = time
    ts = []
    t_hats = []
    distances = []

    # all vs all; compared to the reversed alignment
    if alignment_type == 'combinations':
        for i, j in itertools.combinations(range(n_segments), 2):
            alignment = dtw.dtw(curves[:, i], 
                                curves[:, j], 
                                step_pattern=step_pattern
                                )

            ts.append(np.stack([time[alignment.index2], time[alignment.index1]], 
                                axis=-1
                              )
                     )

            alignment = dtw.dtw(curves[:, j], 
                                curves[:, i], 
                                step_pattern=step_pattern
                               )

            t_hats.append(np.stack([time[alignment.index2], time[alignment.index1]], 
                                   axis=-1
                                  )
                         )

    # all vs all; compared to diagonal
    elif alignment_type == 'permutations':
        for i, j in itertools.permutations(range(n_segments), 2):
            alignment = dtw.dtw(curves[:, i], 
                                curves[:, j], 
                                step_pattern=step_pattern
                               )

            ts.append(np.stack([time[alignment.index2], time[alignment.index1]], 
                               axis=-1
                              )
                     )
            t_hats.append(np.stack([time, time], axis=-1))
    
    # all vs mean; compared to diagonal
    elif alignment_type == 'mean':
        mean_curve = None
        if dba_args is not None:
            mean_curve = dtw_barycentric_averaging(time, curves, **dba_args)
        else:
            mean_curve = dtw_barycentric_averaging(time, curves)

        for i in range(n_segments):
            alignment = dtw.dtw(curves[:, i], 
                                mean_curve, 
                                step_pattern=step_pattern
                                )

            ts.append(np.stack([time[alignment.index2], time[alignment.index1]], 
                               axis=-1
                              )
                     )
            t_hats.append(np.stack([time, time], axis=-1))

    # all vs all; compared to diagonal; dDTW method
    elif alignment_type == 'derivative':
        d_time, d_curves = differentiate(time, curves, method='no_time_symmetric')
        n_samples = d_curves.shape[0]

        for i, j in itertools.permutations(range(n_segments), 2):
            # start by building the distance matrix
            distance_matrix = np.zeros((n_samples, n_samples))
            index_iterator = itertools.product(range(n_samples), repeat=2)
            diff_iterator = itertools.product(d_curves[:, i], d_curves[:, j])

            for (k, l), (d_c_i, d_c_j) in zip(index_iterator, diff_iterator):
                distance_matrix[k, l] = abs(d_c_i - d_c_j)

            # now solve the DTW problem according to these distances
            alignment = dtw.dtw(distance_matrix, 
                                step_pattern=step_pattern
                                )

            # use d_time instead of time (since the numerical derivative cannot 
            # be determined for all time points)
            ts.append(np.stack([d_time[alignment.index2], d_time[alignment.index1]], 
                               axis=-1
                              )
                     )
            t_hats.append(np.stack([d_time, d_time], axis=-1))
        
        new_time = d_time

    # all vs all; compared to diagonal; taking the DTW distance measure
    elif alignment_type == 'distance_measure':
        distances = []
        for i, j in itertools.permutations(range(n_segments), 2):
            alignment = dtw.dtw(curves[:, i], 
                                curves[:, j], 
                                step_pattern=step_pattern
                                )

            distances.append(alignment.distance)

    # all vs all; compared to diagonal; aligning the integrated curves
    elif alignment_type == 'integration':
        distances = []
        for i, j in itertools.permutations(range(n_segments), 2):
            alignment = dtw.dtw(antiderivative(time, curves[:, i]), 
                                antiderivative(time, curves[:, j]), 
                                step_pattern=step_pattern
                                )

            ts.append(np.stack([time[alignment.index2], time[alignment.index1]], 
                                axis=-1
                              )
                     )
            t_hats.append(np.stack([time, time], axis=-1))

    # aligning each segment to a given mean function
    elif alignment_type == 'segmental mean':
        if means_time_tup is None:
            raise ValueError('`mean_time_tup` needs to be specified.')

        mean_curves, common_grid = means_time_tup

        for i in range(n_segments):
            alignment = dtw.dtw(np.interp(common_grid, time, curves[:, i]), 
                                mean_curves[i], 
                                step_pattern=step_pattern
                                )

            ts.append(np.stack([common_grid[alignment.index2], common_grid[alignment.index1]], 
                               axis=-1
                              )
                     )
            t_hats.append(np.stack([common_grid, common_grid], axis=-1))

        new_time = common_grid
        

    else:
        raise ValueError(f'{alignment_type} is not a valid alignment type.')
    

    output = {'time': new_time,
              'time_warping_1': ts,
              'time_warping_2': t_hats,
              'distances': distances
              }

    return output