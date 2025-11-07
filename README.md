# ReSegNet: A Reconstructive Consistency Framework for High-Fidelity Spectrogram Segmentation
Spectrum sensing is a key component in cognitive radio networks, responsible for detecting available spectrum bands and facilitating efficient, adaptive spectrum management. However, existing SOTA segmentation models present critical design flaws. First, a cost-performance trade-off exists: heavyweight models are accurate but prohibitive, while lightweight models are efficient but unreliable. Second, most are passive feed-forward architectures, creating a fragile mapping from noisy input to clean mask that collapses performance in critical, low-SNR environments.

To overcome these limitations, we introduce ReSegNet, a novel framework that fundamentally reframes spectrogram segmentation as a conditional generative reconstruction task. It is tailored for high-fidelity segmentation of coexisting 5G NR, LTE, and radar signals in spectrograms.

