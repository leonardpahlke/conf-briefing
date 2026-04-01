"""Entity canonicalization for CNCF/cloud-native terminology."""

import re

# Curated alias map: lowercase alias → canonical name.
# Covers common abbreviations, misspellings, and alternative names
# across the CNCF landscape and broader cloud-native ecosystem.
CANONICAL_NAMES: dict[str, str] = {
    # Kubernetes
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "kubernetes": "Kubernetes",
    # OpenTelemetry
    "otel": "OpenTelemetry",
    "open telemetry": "OpenTelemetry",
    "opentelemetry": "OpenTelemetry",
    # Prometheus
    "prom": "Prometheus",
    "prometheus": "Prometheus",
    # Argo CD
    "argocd": "Argo CD",
    "argo cd": "Argo CD",
    "argo-cd": "Argo CD",
    # Argo Workflows
    "argo workflows": "Argo Workflows",
    # Istio
    "istio": "Istio",
    # Envoy
    "envoy": "Envoy",
    "envoy proxy": "Envoy",
    # Cilium
    "cilium": "Cilium",
    # eBPF
    "ebpf": "eBPF",
    "e-bpf": "eBPF",
    "bpf": "eBPF",
    # containerd
    "containerd": "containerd",
    # CRI-O
    "crio": "CRI-O",
    "cri-o": "CRI-O",
    # Helm
    "helm": "Helm",
    # Kustomize
    "kustomize": "Kustomize",
    # Flux
    "fluxcd": "Flux",
    "flux cd": "Flux",
    "flux": "Flux",
    # Linkerd
    "linkerd": "Linkerd",
    # Knative
    "knative": "Knative",
    "k-native": "Knative",
    # gRPC
    "grpc": "gRPC",
    # etcd
    "etcd": "etcd",
    # CoreDNS
    "coredns": "CoreDNS",
    "core dns": "CoreDNS",
    # Falco
    "falco": "Falco",
    # Open Policy Agent
    "opa": "Open Policy Agent",
    "open policy agent": "Open Policy Agent",
    # Kyverno
    "kyverno": "Kyverno",
    # Backstage
    "backstage": "Backstage",
    # Crossplane
    "crossplane": "Crossplane",
    # Dapr
    "dapr": "Dapr",
    # KEDA
    "keda": "KEDA",
    # Tekton
    "tekton": "Tekton",
    # Harbor
    "harbor": "Harbor",
    # Rook
    "rook": "Rook",
    # Longhorn
    "longhorn": "Longhorn",
    # Thanos
    "thanos": "Thanos",
    # Cortex
    "cortex": "Cortex",
    # Grafana
    "grafana": "Grafana",
    # Jaeger
    "jaeger": "Jaeger",
    # Zipkin
    "zipkin": "Zipkin",
    # cert-manager
    "cert-manager": "cert-manager",
    "certmanager": "cert-manager",
    "cert manager": "cert-manager",
    # Vitess
    "vitess": "Vitess",
    # SPIFFE/SPIRE
    "spiffe": "SPIFFE",
    "spire": "SPIRE",
    # Buildpacks
    "buildpacks": "Cloud Native Buildpacks",
    "cnb": "Cloud Native Buildpacks",
    "cloud native buildpacks": "Cloud Native Buildpacks",
    # KubeVirt
    "kubevirt": "KubeVirt",
    "kube virt": "KubeVirt",
    # Volcano
    "volcano": "Volcano",
    # Gateway API
    "gateway api": "Gateway API",
    # WebAssembly / Wasm
    "wasm": "WebAssembly",
    "webassembly": "WebAssembly",
    "web assembly": "WebAssembly",
    # Terraform
    "terraform": "Terraform",
    "tf": "Terraform",
    # OpenTofu
    "opentofu": "OpenTofu",
    "open tofu": "OpenTofu",
    # Pulumi
    "pulumi": "Pulumi",
    # Docker
    "docker": "Docker",
    # Podman
    "podman": "Podman",
    # AWS
    "aws": "AWS",
    "amazon web services": "AWS",
    # GCP
    "gcp": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    # Azure
    "azure": "Azure",
    "microsoft azure": "Azure",
    # KubeFlow
    "kubeflow": "Kubeflow",
    "kube flow": "Kubeflow",
    # Velero
    "velero": "Velero",
    # Trivy
    "trivy": "Trivy",
    # Cosign / Sigstore
    "cosign": "Cosign",
    "sigstore": "Sigstore",
    # HAMi (GPU device management)
    "hami": "HAMi",
    "hemi": "HAMi",
    # Perses (observability dashboards)
    "perses": "Perses",
    "persis": "Perses",
    # Kagent (Kubernetes AI agent)
    "kagent": "Kagent",
    "k-agent": "Kagent",
    # Solo.io / Gloo
    "solo.io": "Solo.io",
    "solodial": "Solo.io",
    "gloo": "Gloo",
    # Headlamp
    "headlamp": "Headlamp",
    # KubeVela
    "kubevela": "KubeVela",
    "kube vela": "KubeVela",
    # Karmada
    "karmada": "Karmada",
    # OpenKruise
    "openkruise": "OpenKruise",
    "kruise": "OpenKruise",
    # Strimzi
    "strimzi": "Strimzi",
    # Keptn
    "keptn": "Keptn",
    # Inspektor Gadget
    "inspektor gadget": "Inspektor Gadget",
    # Kubescape
    "kubescape": "Kubescape",
    # Litmuschaos / LitmusChaos
    "litmus": "LitmusChaos",
    "litmuschaos": "LitmusChaos",
    "litmus chaos": "LitmusChaos",
    # vCluster
    "vcluster": "vCluster",
    # Kata Containers
    "kata containers": "Kata Containers",
    "kata": "Kata Containers",
}

# Pre-compile a regex that matches any alias (longest first to avoid partial matches).
# Word-boundary anchored, case-insensitive.
_ALIAS_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(CANONICAL_NAMES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def canonicalize(text: str) -> str:
    """Replace known aliases with canonical names (case-insensitive, word-boundary)."""
    def _replace(match: re.Match) -> str:
        return CANONICAL_NAMES[match.group(0).lower()]

    return _ALIAS_PATTERN.sub(_replace, text)


def canonicalize_analysis(talk: dict) -> dict:
    """Normalize entity names in a talk analysis dict.

    Applies canonicalization to entity-bearing fields:
    - tools_and_projects (list[str])
    - maturity_assessments[].technology
    - relationships[].entity_a / entity_b
    - technology_stance[].technology
    """
    # tools_and_projects
    if "tools_and_projects" in talk:
        talk["tools_and_projects"] = [canonicalize(t) for t in talk["tools_and_projects"]]

    # maturity_assessments
    for entry in talk.get("maturity_assessments", []):
        if "technology" in entry:
            entry["technology"] = canonicalize(entry["technology"])

    # relationships
    for rel in talk.get("relationships", []):
        if "entity_a" in rel:
            rel["entity_a"] = canonicalize(rel["entity_a"])
        if "entity_b" in rel:
            rel["entity_b"] = canonicalize(rel["entity_b"])

    # technology_stance
    for stance in talk.get("technology_stance", []):
        if "technology" in stance:
            stance["technology"] = canonicalize(stance["technology"])

    return talk
