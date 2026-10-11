import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Vectorized Network Components
# ==========================================
class LIFPopulation:
    """
    Vectorized Leaky Integrate-and-Fire neurons.
    Includes intrinsic plasticity (homeostatic threshold adaptation).
    """
    def __init__(self, n_neurons, dt=1.0, tau_m=20.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0):
        self.n = n_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        
        self.v = np.full(n_neurons, v_rest)
        # Dynamic thresholds for homeostasis
        self.v_thresh = np.full(n_neurons, v_thresh) 
        
        self.spikes = np.zeros(n_neurons, dtype=bool)
        self.trace = np.zeros(n_neurons) # Post-synaptic traces
        
        # Homeostasis parameters
        self.target_rate = 5.0   # Target firing rate (Hz)
        self.rate_window = 1000.0 # ms
        self.firing_rates = np.zeros(n_neurons)
        self.tau_thresh = 10000.0 # Slow threshold adaptation

    def step(self, I_input):
        # Check for spikes
        self.spikes = self.v >= self.v_thresh
        
        # Reset spiked neurons
        self.v[self.spikes] = self.v_reset
        self.trace[self.spikes] += 1.0
        
        # Update membrane potential (LIF equation)
        dv = (-self.v + self.v_rest + I_input) / self.tau_m
        self.v += dv * self.dt
        
        # Decay traces
        self.trace *= np.exp(-self.dt / 20.0)
        
        # --- Homeostatic Plasticity (Intrinsic Plasticity) ---
        # Update estimated firing rate (exponential moving average)
        instant_rate = self.spikes * (1000.0 / self.dt) # Convert to Hz
        alpha = self.dt / self.rate_window
        self.firing_rates = (1 - alpha) * self.firing_rates + alpha * instant_rate
        
        # Adjust thresholds to maintain target rate
        # If firing too fast -> increase threshold; if too slow -> decrease
        rate_error = self.firing_rates - self.target_rate
        self.v_thresh += (self.dt / self.tau_thresh) * rate_error
        # Clamp thresholds to biological ranges
        self.v_thresh = np.clip(self.v_thresh, -55.0, -45.0)
        
        return self.spikes

class VectorizedSTDP:
    """
    Efficient Trace-based STDP for a fully connected synaptic matrix.
    W shape: (N_post, N_pre)
    """
    def __init__(self, n_pre, n_post, dt=1.0, tau_plus=20.0, A_plus=0.01, A_minus=0.01, w_max=1.0):
        self.dt = dt
        self.tau_plus = tau_plus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        
        # Initialize weights randomly
        self.W = np.random.uniform(0.1, 0.4, (n_post, n_pre))
        self.x_pre = np.zeros(n_pre) # Pre-synaptic traces
        
    def step(self, pre_spikes, post_traces):
        # Decay pre-synaptic traces
        self.x_pre *= np.exp(-self.dt / self.tau_plus)
        
        # Calculate weight changes
        # LTD: Pre fires -> depression based on post_trace
        # We use broadcasting: pre_spikes (N_pre,) -> (1, N_pre)
        # post_traces (N_post,) -> (N_post, 1)
        dw_ltd = -self.A_minus * post_traces[:, np.newaxis] * pre_spikes[np.newaxis, :]
        
        # Update pre-synaptic trace
        self.x_pre[pre_spikes] += 1.0
        
        # LTP: Post fires -> potentiation based on pre_trace
        # Detected by post_traces being > 0 (we increment by 1 on spike, so we check for new spikes)
        # A cleaner way: pass post_spikes directly.
        # Let's modify to pass post_spikes instead of post_traces for the LTP trigger.
        # Actually, the prompt structure used post_traces. Let's stick to post_traces > threshold.
        post_spiked_mask = post_traces > 0.9 # Incremented by 1.0 in LIF step
        dw_ltp = self.A_plus * self.x_pre[np.newaxis, :] * post_spiked_mask[:, np.newaxis]
        
        # Apply updates and clip weights
        self.W += dw_ltd + dw_ltp
        self.W = np.clip(self.W, 0.0, self.w_max)

# ==========================================
# 2. Simulation Setup
# ==========================================
def run_feature_extraction_sim():
    dt = 1.0
    T = 10000 # 10 seconds
    time = np.arange(0, T, dt)
    
    N_pre = 100
    N_post = 1
    
    # Instantiate network
    pre_pop = LIFPopulation(N_pre, dt=dt)
    post_pop = LIFPopulation(N_post, dt=dt, target_rate=2.0) # Target 2 Hz for sparse coding
    
    synapse = VectorizedSTDP(N_pre, N_post, dt=dt, A_plus=0.005, A_minus=0.005)
    
    # Define a "Pattern": Neurons 20-30 fire at high rate
    pattern_indices = np.arange(20, 31)
    pattern_rate = 30.0 # Hz
    bg_rate = 2.0       # Hz
    
    # Logging
    log_pre_spikes = []
    log_post_spikes = []
    log_W = []
    log_thresh = []
    log_rate = []
    
    # Simulation Loop
    for t in time:
        # 1. Generate Poisson inputs
        # Background noise
        rates = np.full(N_pre, bg_rate)
        # Inject pattern periodically (every 500ms for 200ms)
        if (t % 1000) < 200:
            rates[pattern_indices] = pattern_rate
            
        # Convert rates to spike probabilities (Poisson)
        p_spike = rates * (dt / 1000.0)
        pre_spikes = np.random.rand(N_pre) < p_spike
        
        # 2. Process Pre-synaptic population
        # (We bypass LIF dynamics for pre-population and just use Poisson spikes directly for clarity,
        # but we feed their spikes into the synapse)
        pre_pop.x_pre = np.zeros(N_pre) # reset pre-pop trace, handled by synapse
        
        # 3. Calculate Current to Post-synaptic neuron
        # I = W * S_pre
        I_post = np.dot(synapse.W, pre_spikes.astype(float)) * 2.5 # Scale current
        
        # 4. Process Post-synaptic neuron
        post_spikes = post_pop.step(I_post)
        
        # 5. Update Synapses (STDP)
        synapse.step(pre_spikes, post_pop.trace)
        
        # 6. Logging
        if t % 10 == 0: # Log every 10ms to save memory
            log_pre_spikes.append(pre_spikes.copy())
            log_post_spikes.append(post_spikes.copy())
            log_W.append(synapse.W.copy())
            log_thresh.append(post_pop.v_thresh[0])
            log_rate.append(post_pop.firing_rates[0])

    # ==========================================
    # 3. Visualization
    # ==========================================
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14, 12))
    
    # Convert logs to arrays
    log_pre_spikes = np.array(log_pre_spikes)
    log_post_spikes = np.array(log_post_spikes).squeeze()
    log_W = np.array(log_W).squeeze() # Shape: (Time, N_pre)
    
    # Plot 1: Spike Raster (Pre and Post)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    # Plot pre-spikes
    t_grid, n_grid = np.where(log_pre_spikes.T)
    ax1.scatter(t_grid*10, n_grid, s=2, color='blue', alpha=0.3, label='Pre (Noise+Pattern)')
    # Plot post-spikes
    post_t = np.where(log_post_spikes)[0] * 10
    ax1.scatter(post_t, np.full_like(post_t, N_pre+5), s=20, color='red', marker='|', label='Post (Output)')
    
    # Highlight pattern region
    ax1.axhspan(20, 30, color='green', alpha=0.2, label='Pattern Neurons')
    ax1.set_title('Spike Raster: Network learns to detect the pattern')
    ax1.set_ylabel('Neuron Index')
    ax1.set_xlabel('Time (ms)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-2, N_pre+10)

    # Plot 2: Weight Evolution Matrix
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    # Plot weights at start, middle, and end
    ax2.plot(log_W[0], color='gray', alpha=0.5, label='Initial (t=0)')
    ax2.plot(log_W[len(log_W)//2], color='orange', alpha=0.7, label='Middle (t=5s)')
    ax2.plot(log_W[-1], color='red', linewidth=2, label='Final (t=10s)')
    ax2.axvspan(20, 30, color='green', alpha=0.2)
    ax2.set_title('Synaptic Weight Distribution')
    ax2.set_ylabel('Weight Strength')
    ax2.set_xlabel('Pre-synaptic Neuron Index')
    ax2.legend()

    # Plot 3: Final Weight Heatmap
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    # Reshape for heatmap visualization (just show the 1D array as 2D grid for intuition)
    final_w_grid = log_W[-1].reshape(10, 10)
    im = ax3.imshow(final_w_grid, cmap='hot', aspect='auto')
    ax3.set_title('Final Weights Heatmap (10x10 grid)')
    ax3.set_xticks([])
    ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Plot 4: Homeostatic Threshold Adaptation
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.plot(np.arange(0, T, 10), log_thresh, color='purple')
    ax4.set_title('Homeostatic Plasticity: Firing Threshold Adaptation')
    ax4.set_ylabel('Threshold (mV)')
    ax4.set_xlabel('Time (ms)')

    # Plot 5: Post-synaptic Firing Rate
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(np.arange(0, T, 10), log_rate, color='green')
    ax5.axhline(2.0, color='black', linestyle='--', label='Target Rate (2 Hz)')
    ax5.set_title('Post-synaptic Firing Rate (Stabilizes)')
    ax5.set_ylabel('Firing Rate (Hz)')
    ax5.set_xlabel('Time (ms)')
    ax5.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_feature_extraction_sim()