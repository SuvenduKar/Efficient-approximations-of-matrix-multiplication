"""
=========================================================================================
END-TO-END BENCHMARKING: FIRST-ORDER SVD APPROXIMATION FOR LLM INFERENCE
=========================================================================================
**********WE ARE USING SOME SAMPLE TEXT LINES IN THIS EXPERIMENT.***********
TASK DESCRIPTION:
This script benchmarks the practical speedup and quality retention of the "First-Order 
Approximation of Matrix Multiplication using Truncated Decompositions." It tests the 
method end-to-end on a Large Language Model (LLM) during the "Prefill" phase (processing 
a long input prompt), which relies heavily on large dense matrix multiplications.

COMPONENTS OF THE CODE:
1. FirstOrderLinear (nn.Module): 
   - Replaces standard PyTorch nn.Linear layers.
   - OFFLINE PHASE (__init__): Precomputes the truncated SVD of the static weight matrix
     B to save runtime compute.
   - ONLINE PHASE (forward): Computes the randomized SVD of the dynamic input activations
     A on-the-fly.
   - BLOCK-MATRIX MATH: Executes the first-order approximation: M ≈ (A_approx * B) + (Delta_A * B_approx).
     The multiplications are grouped strategically using associativity (e.g., A @ (B @ C)) 
     to ensure the time complexity never exceeds O(k * n^2), completely avoiding O(n^3) bottlenecks.

2. replace_linear_layers():
   - Recursively traverses the LLM architecture and swaps targeted exact nn.Linear layers 
     (like Q, K, V projections) with our FirstOrderLinear module.

3. benchmark_llm():
   - Loads an open-source model and runs inference on a long batch of text.
   - Evaluates the Unmodified Baseline first.
   - Iteratively evaluates the First-Order method across different values of 's' (proxy for number of components).
   - Measures End-to-End Latency (speed) and Perplexity (accuracy/quality).

4. plot_results():
   - Multiple timed runs with median reporting for better stability.
   - Generates a Pareto front plot comparing the Speed/Accuracy trade-offs of the 
     approximations against the exact baseline.
=========================================================================================


"""
 
import os
#####COMMENT THE BELOW 5 CODE LINES TO ENSURE MUTI-CORE UTILIZATION
# Force strictly sequential (1-core) execution for pure algorithmic benchmarking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
#########

import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from transformers import AutoModelForCausalLM, AutoTokenizer
 

 
 
def randomized_partial_svd(A, s_factor, niter=2):
    """
    Algorithm 2.1 from Kar et al. (2025), with power iterations enabled.
    """
    m, n = A.shape
    k = min(int(s_factor * math.log2(n)) + 1, min(m, n))
 
    orig_dtype = A.dtype
    A_fp32 = A.float()
 
    # Step 4: Random projection
    Omega = torch.randn(n, k, dtype=torch.float32, device=A.device)
 
    # Step 5: Y = A @ Omega
    Y = A_fp32 @ Omega
 
  
 
    # Step 6: QR
    Q, _ = torch.linalg.qr(Y)
 
    # Step 7-10: Project, SVD, lift back
    B = Q.T @ A_fp32
    U_tilde, Sigma, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    V = Vh.T  # torch.linalg.svd returns V^H (== V^T for real matrices)
 
    return U.to(orig_dtype), Sigma.to(orig_dtype), V.to(orig_dtype), k
 
 
class FirstOrderLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using the first-order approximation:
        A @ B  ≈  A_k @ B  +  ΔA @ B_k
    where A = activations (online), B = W^T (offline, precomputed).
    """
 
    def __init__(self, original_linear, s_factor, niter=2):
        super().__init__()
        # PyTorch Linear: out = x @ W^T + bias.  We store B = W^T.
        self.B = original_linear.weight.data.T.clone()
        self.bias = (original_linear.bias.data.clone()
                     if original_linear.bias is not None else None)
        self.s_factor = s_factor
        self.niter = niter
 
        # --- OFFLINE: truncated SVD of the static weight matrix B ---
        U_B, S_B, V_B, self.k_B = randomized_partial_svd(
            self.B, s_factor=self.s_factor, niter=self.niter
        )
        # Store factors for B_k = U_B @ diag(S_B) @ V_B^T
        # Using broadcasting instead of torch.diag:
        #   diag(S_B) @ V_B^T  ==  S_B[:, None] * V_B^T
        self.U_B = U_B                          # (d_in, k_B)
        self.SV_B = S_B.unsqueeze(1) * V_B.T    # (k_B, d_out)
 
    def forward(self, x):
        orig_shape = x.shape
        A = x.reshape(-1, orig_shape[-1])          # (N, d_in)
 
        # --- ONLINE: truncated SVD of dynamic activations A ---
        U_A, S_A, V_A, k_A = randomized_partial_svd(
            A, s_factor=self.s_factor, niter=self.niter
        )
        
        # Use the smaller k to stay within both decompositions
        #k = min(k_A, self.k_B)
        k_B=self.k_B
        U_A = U_A[:, :k_A]                # (N, k)
        S_A = S_A[:k_A]                   # (k,)
        V_A = V_A[:, :k_A]                # (d_in, k)
        U_B = self.U_B[:, :k_B]           # (d_in, k)
        SV_B = self.SV_B[:k_B, :]         # (k, d_out)
 
        # Promote to float32 for the residual arithmetic
        A32 = A.float()
        B32 = self.B.float()
        U_A32, S_A32, V_A32 = U_A.float(), S_A.float(), V_A.float()
        U_B32 = U_B.float()
        SV_B32 = SV_B.float()
 
        # ---------------------------------------------------------------
        # TERM 1:  A_k @ B  =  U_A @ diag(S_A) @ (V_A^T @ B)
        # ---------------------------------------------------------------
        VtB = V_A32.T @ B32                        # (k, d_out)
        SVtB = S_A32.unsqueeze(1) * VtB             # (k, d_out)   [diag scaling]
        term1 = U_A32 @ SVtB                        # (N, d_out)
 
        # ---------------------------------------------------------------
        # TERM 2:  ΔA @ B_k  =  A @ B_k  −  A_k @ B_k
        #   avoids materialising the full N×d_in residual matrix ΔA
        # ---------------------------------------------------------------
        # A @ B_k = A @ U_B @ (SV_B)
        A_UB = A32 @ U_B32                           # (N, k)
        A_Bk = A_UB @ SV_B32                         # (N, d_out)
 
        # A_k @ B_k = U_A @ diag(S_A) @ (V_A^T @ U_B) @ (SV_B)
        VtUB = V_A32.T @ U_B32                       # (k, k)
        SVtUB = S_A32.unsqueeze(1) * VtUB             # (k, k)
        Ak_Bk = U_A32 @ (SVtUB @ SV_B32)             # (N, d_out)
 
        term2 = A_Bk - Ak_Bk                          # (N, d_out)
 
        # ---------------------------------------------------------------
        # Assemble and cast back
        # ---------------------------------------------------------------
        out_flat = (term1 + term2).to(x.dtype)
        out = out_flat.reshape(*orig_shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out
 
 
def replace_linear_layers(model, s_factor, niter=2,
                          target_modules=("q_proj", "k_proj", "v_proj",
                                          "out_proj", "fc1", "fc2")):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, FirstOrderLinear(module, s_factor, niter))
        else:
            replace_linear_layers(module, s_factor, niter, target_modules)
 
 
# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------
 
def timed_forward(model, inputs, n_warmup=1, n_runs=3):
    """Run n_warmup + n_runs forward passes, return median latency (ms) and loss."""
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**inputs, labels=inputs["input_ids"])
 
        latencies = []
        loss_val = None
        for _ in range(n_runs):
            start = time.perf_counter()
            outputs = model(**inputs, labels=inputs["input_ids"])
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            loss_val = outputs.loss
 
    median_lat = float(np.median(latencies))
    perplexity = torch.exp(loss_val).item()
    return median_lat, perplexity
 
 
def benchmark_llm(model_id="facebook/opt-1.3b",
                  prompt_text="We want to see the computation gain of our "
                              "approximation method compared to the baseline.",
                  s_values=(2, 5, 10)):
 
    device = torch.device("cpu")
    print(f"Using device: {device}")
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    single_prompt = prompt_text * 120
    inputs = tokenizer([single_prompt] * 23, return_tensors="pt",
                       truncation=True, max_length=2048).to(device)
 
    seq_len = inputs["input_ids"].shape[1]
    batch   = inputs["input_ids"].shape[0]
    print(f"Input shape: batch={batch}, seq_len={seq_len}  "
          f"(total token-positions = {batch * seq_len})")
 
    results = {"s": [], "perplexity": [], "latency": []}
 
    # ---- BASELINE (exact) ----
    print("\n--- Benchmarking EXACT Baseline ---")
    model_baseline = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_baseline.eval()
 
    base_latency, base_perplexity = timed_forward(model_baseline, inputs)
    print(f"  Latency (median): {base_latency:.2f} ms  |  Perplexity: {base_perplexity:.4f}")
    del model_baseline
    gc.collect()
 
    # ---- FIRST-ORDER APPROXIMATIONS ----
    for s in s_values:
        print(f"\n--- Benchmarking First-Order SVD  s = {s} ---")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()
        replace_linear_layers(model, s_factor=s, niter=2)
 
        lat, ppl = timed_forward(model, inputs)
        print(f"  Latency (median): {lat:.2f} ms  |  Perplexity: {ppl:.4f}")
 
        results["s"].append(s)
        results["latency"].append(lat)
        results["perplexity"].append(ppl)
        del model
        gc.collect()
 
    return results, base_latency, base_perplexity
 
 
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
 
def plot_results(results, base_latency, base_perplexity):

    plt.figure(figsize=(9, 6))
 
    plt.plot(results["latency"], results["perplexity"],
             'k--', zorder=1, alpha=0.4)
 
    markers = {2: 'o', 5: 's', 10: 'D'}
    colors  = {2: 'purple', 5: 'teal', 10: 'gold'}
 
    for i, s in enumerate(results["s"]):
        m = markers.get(s, '^')
        c = colors.get(s, 'blue')
        plt.scatter(results["latency"][i], results["perplexity"][i],
                    color=c, marker=m, s=250, alpha=0.6,
                    edgecolors='black', linewidths=1.5, zorder=5)
 
    plt.scatter([base_latency], [base_perplexity],
                color='red', marker='*', s=450, alpha=0.9,
                edgecolors='black', linewidths=1.5, zorder=10)
 
    legend_handles = [
        mlines.Line2D([], [], color='w', marker='o',
                      markerfacecolor='purple', markeredgecolor='black',
                      markersize=16, alpha=0.6, label='s = 2'),
        mlines.Line2D([], [], color='w', marker='s',
                      markerfacecolor='teal', markeredgecolor='black',
                      markersize=16, alpha=0.6, label='s = 5'),
        mlines.Line2D([], [], color='w', marker='D',
                      markerfacecolor='gold', markeredgecolor='black',
                      markersize=16, alpha=0.6, label='s = 10'),
        mlines.Line2D([], [], color='w', marker='*',
                      markerfacecolor='red', markeredgecolor='black',
                      markersize=20, alpha=0.9, label='Exact Baseline'),
    ]
 
    plt.legend(handles=legend_handles, loc='upper right',
               fontsize=20, framealpha=0.9)
    plt.xlabel('End-to-End Latency (ms)', fontsize=22)
    plt.ylabel('Perplexity', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('llm_test_125m.png', dpi=150)
    plt.show()
 
 
if __name__ == "__main__":
    results, b_lat, b_perp = benchmark_llm(model_id="facebook/opt-125m")
    plot_results(results, b_lat, b_perp)