import numpy as np

def get_length(vec_crystal, at):
    """
    크리스탈 좌표계(Crystal coordinate)의 벡터를 카테시안 좌표계로 변환하여 길이를 구함.
    at: 격자 벡터 행렬 (3x3), 열 벡터가 각 축을 나타냄. Fortran의 at(:, i)와 동일.
    """
    # at 행렬과 크리스탈 벡터 내적 (카테시안 변환)
    vec_cartesian = np.einsum('ij, ...j -> ...i', at, vec_crystal)
    return np.linalg.norm(vec_cartesian, axis=-1)

def get_ws_cutoff(rdim, at, large_cutoff=False):
    """
    Wigner-Seitz cell 탐색 반경(Cutoff) 계산.
    """
    rdim = np.array(rdim)
    if large_cutoff:
        ndim = rdim
    else:
        ndim = rdim // 2 + 1

    i, j, k = np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')
    vecs = np.stack([i.flatten(), j.flatten(), k.flatten()], axis=-1) * ndim

    lengths = get_length(vecs, at)
    return np.max(lengths)

def init_rvec_images(nr, at, large_cutoff=False, ws_search_range=3):
    """
    1단계: Supercell 주기성(+/- 3)을 고려해 cutoff 반경 내의 후보 이미지 수집
    """
    nr = np.array(nr)
    cutoff = get_ws_cutoff(nr, at, large_cutoff)

    # 기본 격자점 (0 ~ nr-1)
    i, j, k = np.mgrid[0:nr[0], 0:nr[1], 0:nr[2]]
    base_vecs = np.stack([i.flatten(), j.flatten(), k.flatten()], axis=-1)

    # 주기성 탐색을 위한 M 벡터 (-3 ~ 3)
    mi, mj, mk = np.mgrid[-ws_search_range:ws_search_range+1,
                          -ws_search_range:ws_search_range+1,
                          -ws_search_range:ws_search_range+1]
    shifts = np.stack([mi.flatten(), mj.flatten(), mk.flatten()], axis=-1) * nr # (343, 3)

    images_dict = {}

    # 격자점별 후보군 추리기
    for ir, vec in enumerate(base_vecs):
        rvecs = vec + shifts  # (343, 3)
        lengths = get_length(rvecs, at)
        valid_mask = lengths < cutoff

        valid_rvecs = rvecs[valid_mask]
        if len(valid_rvecs) < 1:
            raise ValueError(f"nim < 1 error: {vec} 위치의 이미지를 못 찾음.")

        images_dict[ir] = valid_rvecs

    return images_dict

def set_wigner_seitz_cell(nr_or_images, at, tau_a, tau_b, eps6=1e-6, large_cutoff=False):
    """
    2단계: 최단 거리 Wigner-Seitz cell 벡터(rvec)와 겹침수(ndeg) 찾기.
    입력:
        nr_or_images: [nr1, nr2, nr3] 그리드 크기 리스트 OR init_rvec_images에서 반환된 images_dict
        at: (3, 3) 격자 벡터
        tau_a, tau_b: 두 대상의 크리스탈 좌표 (예: 전자 wannier 중심, 원자 위치)
    반환:
        ws_rvec: 최단거리 WS 벡터 배열 (M, 3)
        ws_ndeg: 겹침 수 배열 (M,)
    """
    if isinstance(nr_or_images, dict):
        images_dict = nr_or_images
    else:
        images_dict = init_rvec_images(nr_or_images, at, large_cutoff)

    ws_rvec_list = []
    ws_ndeg_list = []

    tau_a = np.array(tau_a)
    tau_b = np.array(tau_b)

    for ir, rvecs in images_dict.items():
        # rvecs 모양: (nim, 3)
        vec_b = rvecs + tau_b
        # 거리: | (R + tau_b) - tau_a |
        dist = get_length(vec_b - tau_a, at)

        # 최단 거리
        min_dist = np.min(dist)

        # 오차(eps6) 내의 최단거리 이미지들
        is_shortest = (dist - min_dist) < eps6
        shortest_rvecs = rvecs[is_shortest]
        ndeg = np.sum(is_shortest)

        if ndeg < 1:
            raise ValueError("ndeg < 1 error: 최단거리 벡터 산출 실패.")

        for rvec in shortest_rvecs:
            ws_rvec_list.append(rvec)
            ws_ndeg_list.append(ndeg)

    return np.array(ws_rvec_list), np.array(ws_ndeg_list)
