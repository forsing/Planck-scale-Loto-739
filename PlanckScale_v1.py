# Potreban je NumPy ≥ 2.0



import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CSV_PATH = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
    # "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
    # "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"
)


N = 39
K = 7

STRENGTHS = (
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)

FOLD_FRACTIONS = (0.5, 0.65, 0.8, 1.0)


@dataclass
class Region:
    low: int
    high: int
    depth: int = 0
    left: object = None
    right: object = None
    index: int = -1

    @property
    def size(self):
        return self.high - self.low + 1


def make_tree(
    low=1,
    high=N,
    depth=0,
    internal=None,
    root_cut=None,
):
    if internal is None:
        internal = []

    node = Region(low, high, depth)

    if low < high:
        midpoint = (
            root_cut
            if depth == 0 and root_cut is not None
            else (low + high) // 2
        )

        node.index = len(internal)
        internal.append(node)

        node.left = make_tree(
            low,
            midpoint,
            depth + 1,
            internal,
        )
        node.right = make_tree(
            midpoint + 1,
            high,
            depth + 1,
            internal,
        )

    return node


def make_forest(n=N):
    """
    Tri determinističke hijerarhije.

    Različite grube podele smanjuju zavisnost modela
    od jedne proizvoljne granice opsega.
    """
    result = []

    cuts = sorted(set((
        n // 3,
        (n + 1) // 2,
        2 * n // 3,
    )))

    for cut in cuts:
        nodes = []

        root = make_tree(
            1,
            n,
            internal=nodes,
            root_cut=cut,
        )

        result.append((root, nodes))

    return result


def load_csv(path):
    """
    CSV bez zaglavlja, sedam brojeva po redu.
    Prvi red najstariji, poslednji najnoviji.
    """
    draws = []

    with Path(path).open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for line, row in enumerate(csv.reader(file), 1):
            if not row or all(
                not value.strip() for value in row
            ):
                continue

            try:
                draw = sorted(
                    int(value.strip()) for value in row
                )
            except ValueError as exc:
                raise ValueError(
                    f"Red {line}: neispravan ceo broj."
                ) from exc

            if (
                len(draw) != K
                or len(set(draw)) != K
                or not all(
                    1 <= value <= N for value in draw
                )
            ):
                raise ValueError(
                    f"Red {line}: potrebno je sedam "
                    "različitih brojeva od 1 do 39."
                )

            draws.append(draw)

    if len(draws) < 50:
        raise ValueError(
            "Potrebno je najmanje 50 izvlačenja."
        )

    return np.asarray(draws, dtype=np.int64)


def prefix_counts(draws, n=N):
    indicator = np.zeros(
        (len(draws), n),
        dtype=np.int64,
    )

    indicator[
        np.arange(len(draws))[:, None],
        draws - 1,
    ] = 1

    return np.column_stack((
        np.zeros(len(draws), dtype=np.int64),
        np.cumsum(indicator, axis=1),
    ))


def observations(prefix, node):
    total = (
        prefix[:, node.high]
        - prefix[:, node.low - 1]
    )

    left = (
        prefix[:, node.left.high]
        - prefix[:, node.low - 1]
    )

    return total, left


def learn_statistics(prefix, nodes):
    """
    Statistike zajedničkih rasporeda kroz hijerarhiju.

    Ne rangiraju se frekvencije pojedinačnih brojeva.
    Uče se uslovne distribucije raspodele sedmorke
    između povezanih oblasti.
    """
    result = []

    for node in nodes:
        total, left = observations(prefix, node)

        counts = np.zeros(
            (K + 1, K + 1),
            dtype=np.int64,
        )

        np.add.at(counts, (total, left), 1)
        result.append(counts)

    return result


def distributions(statistics, nodes, strength):
    """
    Normalizovane distribucije podele.

    Kombinatorni prior uvažava izvlačenje bez ponavljanja.
    Regularizacija raste na finim skalama.

    Dodatno skaliranje obezbeđuje minimalnu podršku
    svakom dozvoljenom rasporedu, kako retke oblasti
    ne bi dobijale preveliku prednost iz malo podataka.
    """
    result = []

    for node, counts in zip(nodes, statistics):
        left_size = node.left.size
        right_size = node.right.size

        q = np.zeros_like(counts, dtype=float)
        concentration = strength * (node.depth + 1)

        for total in range(min(K, node.size) + 1):
            lower = max(0, total - right_size)
            upper = min(total, left_size)
            values = np.arange(lower, upper + 1)

            prior = np.asarray([
                math.comb(left_size, int(value))
                * math.comb(
                    right_size,
                    total - int(value),
                )
                / math.comb(node.size, total)
                for value in values
            ])

            row = counts[total, values]

            local_concentration = (
                concentration / float(prior.min())
            )

            q[total, values] = (
                row + local_concentration * prior
            ) / (
                row.sum() + local_concentration
            )

            if not np.isclose(
                q[total].sum(),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Distribucija podele nije normalizovana."
                )

        result.append(q)

    return result


def log_probabilities(prefix, nodes, q):
    result = np.zeros(len(prefix))

    for node in nodes:
        total, left = observations(prefix, node)
        probability = q[node.index][total, left]

        if np.any(probability <= 0.0):
            raise RuntimeError(
                "Neispravna verovatnoća "
                "validne kombinacije."
            )

        result += np.log(probability)

    return result


def mixture_log_probabilities(prefix, forest, q_forest):
    """
    Zajednička distribucija je jednaka mešavina
    tri normalizovane višeskalne distribucije.
    """
    parts = np.asarray([
        log_probabilities(prefix, nodes, q)
        for (_, nodes), q in zip(forest, q_forest)
    ])

    return (
        np.logaddexp.reduce(parts, axis=0)
        - math.log(len(forest))
    )


def choose_strength(prefix, forest):
    """
    Tri hronološke provere.

    Svaki kasniji blok ocenjuje se modelom naučenim
    samo na ranijim redovima.
    """
    boundaries = [
        int(value * len(prefix))
        for value in FOLD_FRACTIONS
    ]

    cache = []

    for start, stop in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        stats = [
            learn_statistics(prefix[:start], nodes)
            for _, nodes in forest
        ]

        cache.append((stats, prefix[start:stop]))

    best = None
    best_score = -math.inf

    print(
        "Hronološka provera: "
        f"{len(prefix) - boundaries[0]} "
        "kasnijih izvlačenja.",
        flush=True,
    )

    print(
        "Referentni log P: "
        f"{-math.log(math.comb(N, K)):.9f}",
        flush=True,
    )

    for strength in STRENGTHS:
        total_score = 0.0
        rows = 0

        for stats, validation in cache:
            qs = [
                distributions(statistics, nodes, strength)
                for statistics, (_, nodes)
                in zip(stats, forest)
            ]

            total_score += float(
                mixture_log_probabilities(
                    validation,
                    forest,
                    qs,
                ).sum()
            )

            rows += len(validation)

        score = total_score / rows

        print(
            f"Regularizacija={strength:g}; "
            f"provera log P={score:.9f}",
            flush=True,
        )

        if score > best_score:
            best = strength
            best_score = score

    return best, best_score


def maximum_and_normalization(node, q, cache=None):
    """
    Maksimum i normalizacija jedne hijerarhije
    za svaki dozvoljeni broj izabranih brojeva.
    """
    if cache is None:
        cache = {}

    capacity = min(K, node.size)

    scores = np.full(capacity + 1, -np.inf)
    combinations = [None] * (capacity + 1)
    normalization = np.zeros(capacity + 1)

    if node.left is None:
        scores[:] = 0.0
        combinations[0] = ()
        combinations[1] = (node.low,)
        normalization[:] = 1.0

        cache[(node.low, node.high)] = scores

        return scores, combinations, normalization

    (
        left_score,
        left_combination,
        left_normalization,
    ) = maximum_and_normalization(
        node.left,
        q,
        cache,
    )

    (
        right_score,
        right_combination,
        right_normalization,
    ) = maximum_and_normalization(
        node.right,
        q,
        cache,
    )

    for total in range(capacity + 1):
        lower = max(0, total - node.right.size)
        upper = min(total, node.left.size)

        for left_count in range(lower, upper + 1):
            right_count = total - left_count
            split_probability = q[node.index][
                total,
                left_count,
            ]

            normalization[total] += (
                split_probability
                * left_normalization[left_count]
                * right_normalization[right_count]
            )

            score = (
                math.log(split_probability)
                + left_score[left_count]
                + right_score[right_count]
            )

            candidate = (
                left_combination[left_count]
                + right_combination[right_count]
            )

            if (
                score > scores[total]
                or (
                    score == scores[total]
                    and (
                        combinations[total] is None
                        or candidate < combinations[total]
                    )
                )
            ):
                scores[total] = score
                combinations[total] = candidate

    if not np.allclose(
        normalization,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Distribucije između skala nisu dosledne."
        )

    cache[(node.low, node.high)] = scores

    return scores, combinations, normalization


def fixed_log_probability(node, q, prefix):
    """
    Verovatnoća potpuno poznatog rasporeda
    unutar jedne oblasti.
    """
    total = int(
        prefix[node.high] - prefix[node.low - 1]
    )

    if total == 0 or total == node.size:
        return 0.0

    left = int(
        prefix[node.left.high] - prefix[node.low - 1]
    )

    return (
        math.log(q[node.index][total, left])
        + fixed_log_probability(node.left, q, prefix)
        + fixed_log_probability(node.right, q, prefix)
    )


def constrained_maximum(
    node,
    q,
    cache,
    prefix,
    cutoff,
):
    """
    Najbolji mogući nastavak uz već određene brojeve.

    Brojevi do cutoff imaju fiksirano prisustvo.
    Viši brojevi ostaju slobodni.
    """
    if node.low > cutoff:
        return cache[(node.low, node.high)]

    capacity = min(K, node.size)
    result = np.full(capacity + 1, -np.inf)

    if node.high <= cutoff:
        total = int(
            prefix[node.high] - prefix[node.low - 1]
        )

        result[total] = fixed_log_probability(
            node,
            q,
            prefix,
        )

        return result

    left = constrained_maximum(
        node.left,
        q,
        cache,
        prefix,
        cutoff,
    )

    right = constrained_maximum(
        node.right,
        q,
        cache,
        prefix,
        cutoff,
    )

    for total in range(capacity + 1):
        low = max(0, total - node.right.size)
        high = min(total, node.left.size)
        values = np.arange(low, high + 1)

        result[total] = np.max(
            np.log(q[node.index][total, values])
            + left[values]
            + right[total - values]
        )

    return result


def mixture_maximum(forest, qs, n=N, k=K):
    """
    Globalni maksimum cele mešavine.

    Branch-and-bound koristi matematičku gornju granicu
    iz maksimuma svih hijerarhija pod istim ograničenjem.

    Nema random uzorkovanja ni beam search-a.
    """
    caches = []
    seeds = set()

    for (root, _), q in zip(forest, qs):
        cache = {}

        _, combinations, normalization = (
            maximum_and_normalization(root, q, cache)
        )

        if not np.isclose(
            normalization[k],
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Normalizacija nije prošla."
            )

        seeds.add(combinations[k])
        caches.append(cache)

    best = None
    best_p = -1.0

    for seed in sorted(seeds):
        logp = float(
            mixture_log_probabilities(
                prefix_counts(np.asarray([seed]), n),
                forest,
                qs,
            )[0]
        )

        probability = math.exp(logp)

        if probability > best_p:
            best = seed
            best_p = probability

    def upper_bound(selected):
        indicator = np.zeros(n, dtype=np.int64)

        if selected:
            indicator[np.asarray(selected) - 1] = 1

        prefix = np.concatenate((
            [0],
            np.cumsum(indicator),
        ))

        cutoff = selected[-1] if selected else 0

        scores = [
            constrained_maximum(
                root,
                q,
                cache,
                prefix,
                cutoff,
            )[k]
            for (root, _), q, cache
            in zip(forest, qs, caches)
        ]

        return float(np.exp(scores).mean())

    queue = [(-upper_bound(()), ())]
    visited = 0

    while queue:
        negative_bound, selected = heapq.heappop(queue)
        bound = -negative_bound

        if bound < best_p * (1.0 - 1e-12):
            continue

        visited += 1

        if len(selected) == k:
            if (
                bound > best_p
                or (
                    bound == best_p
                    and selected < best
                )
            ):
                best = selected
                best_p = bound

            continue

        last = selected[-1] if selected else 0
        maximum_value = n - (k - len(selected) - 1)

        for value in range(last + 1, maximum_value + 1):
            child = selected + (value,)
            upper = upper_bound(child)

            if upper >= best_p * (1.0 - 1e-12):
                heapq.heappush(queue, (-upper, child))

    direct = math.exp(float(
        mixture_log_probabilities(
            prefix_counts(np.asarray([best]), n),
            forest,
            qs,
        )[0]
    ))

    if not np.isclose(
        direct,
        best_p,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise RuntimeError(
            "Globalni maksimum nije prošao proveru."
        )

    return best, direct, visited


def main():
    draws = load_csv(CSV_PATH)
    forest = make_forest()
    prefix = prefix_counts(draws)

    print(
        f"Planck scale v1; svih {len(draws)} "
        "izvlačenja; tri hijerarhije.",
        flush=True,
    )

    strength, _ = choose_strength(prefix, forest)

    # Završna obuka koristi CEO ažurni CSV.
    qs = [
        distributions(
            learn_statistics(prefix, nodes),
            nodes,
            strength,
        )
        for _, nodes in forest
    ]

    prediction, probability, visited = mixture_maximum(
        forest,
        qs,
    )

    print(
        f"Završna obuka: svih {len(draws)}; "
        f"regularizacija={strength:g}; "
        f"čvorova={visited}.",
        flush=True,
    )

    print("\nNEXT:", " ".join(map(str, prediction)))
    print(f"Verovatnoća po modelu: {probability:.12g}")


if __name__ == "__main__":
    main()



"""
Planck scale v1; svih 4698 izvlačenja; tri hijerarhije.
Hronološka provera: 2349 kasnijih izvlačenja.
Referentni log P: -16.548639443
Regularizacija=0.1; provera log P=-16.562424302
Regularizacija=1; provera log P=-16.558041832
Regularizacija=10; provera log P=-16.552856453
Regularizacija=100; provera log P=-16.548340076
Regularizacija=1000; provera log P=-16.548323522
Regularizacija=10000; provera log P=-16.548599086
Regularizacija=100000; provera log P=-16.548635303
Završna obuka: svih 4698; regularizacija=1000; čvorova=43.

NEXT: 8 11 23 29 32 34 39
Verovatnoća po modelu: 6.88778501441e-08





Planck scale v1; svih 2971 izvlačenja; tri hijerarhije.
Hronološka provera: 1486 kasnijih izvlačenja.
Referentni log P: -16.548639443
Regularizacija=0.1; provera log P=-16.572847188
Regularizacija=1; provera log P=-16.565921468
Regularizacija=10; provera log P=-16.557269931
Regularizacija=100; provera log P=-16.549649132
Regularizacija=1000; provera log P=-16.548580612
Regularizacija=10000; provera log P=-16.548629185
Regularizacija=100000; provera log P=-16.548638367
Završna obuka: svih 2971; regularizacija=1000; čvorova=8.

NEXT: 8 16 19 23 24 33 38
Verovatnoća po modelu: 6.81975889189e-08





Planck scale v1; svih 1727 izvlačenja; tri hijerarhije.
Hronološka provera: 864 kasnijih izvlačenja.
Referentni log P: -16.548639443
Regularizacija=0.1; provera log P=-16.599863853
Regularizacija=1; provera log P=-16.583808442
Regularizacija=10; provera log P=-16.565272931
Regularizacija=100; provera log P=-16.551157555
Regularizacija=1000; provera log P=-16.548844548
Regularizacija=10000; provera log P=-16.548658909
Regularizacija=100000; provera log P=-16.548641379
Završna obuka: svih 1727; regularizacija=100000; čvorova=35.

NEXT: 8 11 18 23 29 34 37
Verovatnoća po modelu: 6.50397145296e-08
"""



"""
Model distribucije sa više nivoa detalja.
Za loto 7/39 to bi značilo:
- Model uči zajedničku distribuciju preko pojedinačnih brojeva, parova, trojki i složenijih zavisnosti.
- Hronološka provera određuje koliko detaljna distribucija ima podršku u podacima.
- Složenije zavisnosti koje nemaju dovoljno podrške ublažavaju se regularizacijom, da model ne pretvori slučajne istorijske obrasce u jake predikcije.
- Završni model koristi ceo CSV i deterministički bira jednu najverovatniju sedmorku.
Korisna analogija je granica rezolucije modela: matematički mogu opisati veoma detaljne obrasce, ali CSV možda ne pruža dovoljno informacija da ih pouzdano procenim.
Odredjujem koliko detaljnu distribuciju podaci opravdavaju. 


Opis sistema na različitim skalama i povezivanja tih opisa. Planckova skala označava fizički režim u kojem očekujem značaj kvantne gravitacije; višeskalno modelovanje je šira matematička ideja koju mogu preneti na loto.
Višeskalni model zajedničke distribucije 7/39:
- Detaljni nivo: zavisnosti između konkretnih brojeva, parova i trojki.
- Grublji nivo: distribucija strukture cele sedmorke — razmaci, raspon i raspored brojeva po grupama.
- Povezivanje nivoa: verovatnoća grubljeg obrasca mora biti jednaka zbiru verovatnoća svih sedmorki koje mu pripadaju.
- Izbor rezolucije: hronološka provera određuje koji detalji poboljšavaju predikciju, a koji samo pamte istoriju.
Ceo CSV služi za učenje, bez nametanja normalne raspodele i bez random uzorkovanja. Iz završne distribucije bira se jedna sedmorka sa najvećom verovatnoćom.
Dosledna distribucija na više skala. Planckovu dužinu ili energiju ne bih proizvoljno ubacivao u račun; korisna je matematička ideja odnosa između skala.


Višeskalni model gradi zajedničku distribuciju sedmorki od grubih podela opsega 1-39 do pojedinačnih brojeva, tako da verovatnoće budu dosledne između skala. 
Planckova skala ovde daje ideju povezivanja skala. 


Višeskalni statistički model inspirisan povezivanjem skala. Uči zajedničku distribuciju rasporeda sedmorke kroz hijerarhiju opsega. 
Nema normalne raspodele, random postupaka ni rangiranja najčešćih brojeva.
Provereni su normalizacija, doslednost između skala i globalni maksimum. Hronološka provera izabrala je regularizaciju 10000, uz malu prednost nad referencom.
Popravljena zavisnost modela od jedne fiksne podele opsega. Kombinovati više višeskalnih hijerarhija i učiti kako rasporedi u susednim oblastima zavise jedni od drugih.
Ispravljen kod je izvršen na svih 4.698 izvlačenja. 
Sada kombinuje tri višeskalne hijerarhije i jače regularizuje retke rasporede. Hronološka provera izabrala je regularizaciju 1000. Provereni su normalizacija i globalni maksimum zajedničke distribucije.
"""
