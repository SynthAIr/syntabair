import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/airport_network_visualizations")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SIMILARITY_THRESHOLD = 0.9
MIN_AIRPORT_FLIGHTS = 10
MAX_CLUSTERS = 5
DEFAULT_SAMPLE_SIZE = 500_000


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TabSyn embedding visualizations for flight data"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold for network edges (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    return parser.parse_args()


def get_custom_color_palette(num_colors: int) -> List[Tuple[float, float, float]]:
    base_palette = [
        (0.00, 0.63, 0.64),  # Teal
        (0.75, 0.12, 0.24),  # Crimson
        (0.85, 0.37, 0.01),  # Dark orange
        (0.13, 0.55, 0.13),  # Forest green
        (0.00, 0.21, 0.41),  # Navy
        (0.55, 0.00, 0.26),  # Burgundy
        (0.80, 0.52, 0.25),  # Ochre
        (0.64, 0.08, 0.18),  # Dark red
        (0.00, 0.39, 0.25),  # Dark green
        (0.55, 0.27, 0.07),  # Brown
        (0.28, 0.24, 0.55),  # Dark slate
    ]
    if num_colors <= len(base_palette):
        return base_palette[:num_colors]

    # Fallback to matplotlib's tab20 for extra colors
    import seaborn as sns  # lazy import
    extra = sns.color_palette("tab20", num_colors - len(base_palette))
    return base_palette + extra


def load_data(
    embeddings_path: Path,
    data_path: Path,
    sample_size: Optional[int],
    seed: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    logging.info(f"Loading embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        logging.info("Reshaped embeddings to %s", embeddings.shape)

    logging.info("Loading flight data from %s", data_path)
    df = pd.read_csv(data_path)
    df = preprocess_flight_data_for_prediction(df)

    if len(df) != len(embeddings):
        logging.warning(
            "Data length mismatch: flights=%d, embeddings=%d",
            len(df), len(embeddings)
        )
        n = min(len(df), len(embeddings))
        df, embeddings = df.iloc[:n], embeddings[:n]

    if sample_size and sample_size < len(embeddings):
        np.random.seed(seed)
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        df = df.iloc[idx].reset_index(drop=True)
        logging.info("Sampled %d records", sample_size)

    metadata = {"airport": df["DEPARTURE_IATA_AIRPORT_CODE"].values}
    return embeddings, metadata


def build_graph(
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
    threshold: float
) -> Tuple[nx.Graph, List[str], np.ndarray]:
    airports = np.unique(metadata["airport"])
    embed_map = {
        a: embeddings[metadata["airport"] == a].mean(axis=0)
        for a in airports
        if (metadata["airport"] == a).sum() >= MIN_AIRPORT_FLIGHTS
    }
    codes = list(embed_map)
    matrix = np.vstack([embed_map[a] for a in codes])
    sim = cosine_similarity(matrix)

    G = nx.Graph()
    for code in codes:
        size = int((metadata["airport"] == code).sum())
        G.add_node(code, size=size)
    for i, u in enumerate(codes):
        for j, v in enumerate(codes[i+1:], i+1):
            if sim[i, j] > threshold:
                G.add_edge(u, v, weight=sim[i, j])
    return G, codes, sim


def detect_communities(
    G: nx.Graph
) -> Tuple[Dict[str, int], Optional[ListedColormap]]:
    n_nodes = G.number_of_nodes()
    if n_nodes <= 5:
        return {n: 0 for n in G}, None

    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 1.0)
    n_clusters = min(MAX_CLUSTERS, n_nodes)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=DEFAULT_RANDOM_SEED,
    )
    labels = sc.fit_predict(A)
    mapping = dict(zip(G.nodes, labels))
    palette = get_custom_color_palette(len(set(labels)))
    cmap = ListedColormap(palette)
    return mapping, cmap


def draw_and_save(
    G: nx.Graph,
    community_map: Dict[str, int],
    cmap: Optional[ListedColormap],
    sim_matrix: np.ndarray,
    codes: List[str],
    output_dir: Path
) -> None:
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=DEFAULT_RANDOM_SEED, k=0.3)
    sizes = [G.nodes[n]["size"] / 10 + 100 for n in G]
    widths = [G[u][v]["weight"] * 3 for u, v in G.edges]

    colors = [community_map[n] for n in G] if cmap else "skyblue"
    nx.draw_networkx_nodes(
        G, pos, node_size=sizes, node_color=colors,
        cmap=cmap, alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title("Airport Network via TabSyn Embeddings", fontsize=18)
    plt.axis('off')
    if cmap:
        # Legend
        legend_elems = []
        for comm in sorted(set(community_map.values())):
            legend_elems.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(comm), markersize=10,
                       label=f"Community {comm + 1}")
            )
        plt.legend(
            handles=legend_elems,
            title="Communities",
            loc='upper right',
            bbox_to_anchor=(1.05, 1)
        )
    plt.tight_layout()

    img_path = output_dir / "airport_network.png"
    csv_base = img_path.stem
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSVs
    sim_df = pd.DataFrame(sim_matrix, index=codes, columns=codes)
    sim_df.to_csv(output_dir / f"{csv_base}_similarity.csv")

    node_df = pd.DataFrame([
        {"Airport": n, "Flight_Count": G.nodes[n]["size"], "Community": community_map.get(n, 0)}
        for n in G.nodes
    ])
    node_df.to_csv(output_dir / f"{csv_base}_nodes.csv", index=False)

    if G.number_of_edges() > 0:
        edge_data = [
            {"Airport1": u, "Airport2": v, "Similarity": G[u][v]["weight"]}
            for u, v in G.edges
        ]
        pd.DataFrame(edge_data).to_csv(
            output_dir / f"{csv_base}_edges.csv", index=False
        )

    logging.info("Saved visualization and CSVs to %s", output_dir)


def main():
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    G, codes, sim = build_graph(
        embeddings, metadata, args.threshold
    )
    community_map, cmap = detect_communities(G)
    draw_and_save(
        G, community_map, cmap, sim, codes, args.output_dir
    )


if __name__ == "__main__":
    main()
