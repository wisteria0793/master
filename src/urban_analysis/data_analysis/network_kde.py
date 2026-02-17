import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Union, Optional

def quartic_kernel(d: float, bandwidth: float) -> float:
    """
    Quartic（四次）カーネル関数。
    
    Args:
        d (float): 距離。
        bandwidth (float): バンド幅。
        
    Returns:
        float: カーネル値。
    """
    if d > bandwidth:
        return 0.0
    return (1 - (d / bandwidth)**2)**2

def network_kernel_density(
    G: nx.Graph, 
    event_points: pd.DataFrame, 
    bandwidth: float,
    kernel: str = 'quartic',
    node_column: str = 'osm_node'
) -> Dict[Tuple[int, int, int], float]:
    """
    ノードベースの近似を使用して、グラフ上でネットワークカーネル密度推定（NKDE）を計算します。
    
    この関数は、ネットワークエッジに沿ったイベントポイントの密度を計算します。
    各イベントポイントの影響をバンド幅内の近くのノードに伝播させ、
    エッジ密度をその端点の密度から導出するという最適化された手法を使用しています。
    
    Args:
        G (nx.Graph): 道路ネットワークグラフ（通常はOSMnxから取得）。
        event_points (pd.DataFrame): イベントポイントを含むDataFrame。
        bandwidth (float): カーネル密度推定のバンド幅（メートル単位）。
        kernel (str, optional): 使用するカーネル関数。現在は 'quartic' のみがサポートされています。
        node_column (str, optional): 最も近いOSMノードIDを含むevent_points内の列名。
                                     デフォルトは 'osm_node' です。

    Returns:
        Dict[Tuple[int, int, int], float]: エッジキー (u, v, k) を密度値にマッピングする辞書。
                                           密度は強度を表し、エッジの長さで正規化されています。
    """
    if node_column not in event_points.columns:
        raise ValueError(f"event_points DataFrame must contain a '{node_column}' column.")

    # 1. ノード密度の初期化
    node_densities = {node: 0.0 for node in G.nodes()}

    # 2. イベントポイントを反復処理し、ノードへの影響を累積する
    # これは「最適化された」部分です：エッジを反復してイベントを見つけるのではなく、
    # イベントを反復して影響を受けるノードを見つけます。
    
    # ループ内のエラーを避けるためにノードが存在するか事前にチェック
    # G.nodes()のセットに対してisinを使用するのは、行ごとにG.has_nodeを適用するよりもはるかに高速です
    valid_nodes = set(G.nodes())
    valid_mask = event_points[node_column].isin(valid_nodes)
    valid_events = event_points[valid_mask]
    
    # 同じノードに複数のイベントがある場合にDijkstraの呼び出しを最小限に抑えるために、
    # ユニークなイベントノードに対して反復処理を行います。
    # ただし、各ノードのイベント数を考慮する必要があります。
    # ノードごとにグループ化してイベント数をカウントします。
    event_counts = valid_events[node_column].value_counts()
    
    for event_node, count in tqdm(event_counts.items(), total=len(event_counts), desc="NKDE Calculation"):
        # イベントノードからバンド幅内のすべてのノードまでの最短パスを計算
        # Dijkstra法を使用。エッジに'length'属性が存在することを前提としています。
        reachable_nodes_dist = nx.single_source_dijkstra_path_length(
            G, source=event_node, cutoff=bandwidth, weight='length'
        )
        
        # カーネル値を累積
        for node, dist in reachable_nodes_dist.items():
            if kernel == 'quartic':
                k_val = quartic_kernel(dist, bandwidth)
            else:
                # 不明な場合はQuarticにフォールバック、または他を実装
                k_val = quartic_kernel(dist, bandwidth)
            
            # 同じノードにある複数のイベントは同じように寄与するため、カウントを掛けます
            node_densities[node] += k_val * count

    # 3. エッジ密度の計算
    edge_densities = {}
    
    # MultiGraph と Graph の処理
    is_multigraph = G.is_multigraph()
    if is_multigraph:
        edges_iter = G.edges(keys=True, data=True)
    else:
        edges_iter = G.edges(data=True)

    for edge in edges_iter:
        if is_multigraph:
            u, v, k, data = edge
            edge_key = (u, v, k)
        else:
            u, v, data = edge
            # 単純グラフの場合、一貫したフォーマットを維持するためにキーとして0を使用できます
            edge_key = (u, v, 0)
            
        edge_len = data.get('length', 1.0)
        # ゼロ除算の回避
        if edge_len <= 0:
            edge_len = 1.0
            
        u_density = node_densities.get(u, 0.0)
        v_density = node_densities.get(v, 0.0)
        
        # エッジ密度はその端点密度の平均です
        # エッジ長で正規化して「線密度」（メートルあたりの強度）を表します
        # この正規化は、レガシーな「最適化された」実装にも存在していました。
        avg_node_density = (u_density + v_density) / 2.0
        edge_density = avg_node_density / edge_len
        
        edge_densities[edge_key] = edge_density

    return edge_densities
