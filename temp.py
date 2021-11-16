#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def dist_selector(c_coord, points, n_points):
    """some docstring"""
    dist_array = distance.cdist([c_coord], points,
                                'euclidean')[0]
    dist_index = np.argsort(dist_array)
    selected_dists = dist_array[dist_index][0:n_points]
    selected_coords = points[dist_index][0:n_points]
    return selected_coords, selected_dists, dist_index
