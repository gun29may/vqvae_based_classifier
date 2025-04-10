# VQVAE-Based Classifier

This project implements a VQVAE (Vector Quantized Variational Autoencoder) based classifier.  The goal is to learn a compact, discrete representation of input data using a VQVAE, and then leverage this representation for classification.

## Methodology

The project follows a two-stage training approach:

1.  **VQVAE Training:** A VQVAE is trained to reconstruct input data. The key component of a VQVAE is the codebook (or embedding space), which forces the encoder to map inputs to a discrete set of vectors.  This encourages the network to learn a more structured and interpretable representation compared to continuous latent spaces in standard VAEs.  The training objective minimizes reconstruction loss and a commitment loss that encourages the encoder output to "commit" to a codebook entry.

2.  **Classifier Training:** After training the VQVAE, its encoder is frozen. A classification head is then trained using the frozen encoder's output as input.  The idea is that the VQVAE encoder has learned a useful representation of the data, and this representation can be used to effectively train a classifier, even with potentially limited labeled data.

## Code Structure

The project code is organized into several modules for flexibility and extensibility:

*   **`models/encoder.py`:** Contains the encoder architecture for the VQVAE.
*   **`models/decoder.py`:** Contains the decoder architecture for the VQVAE.
*   **`models/quantizer.py`:** Implements the vector quantization logic, including the codebook and nearest-neighbor lookup.
*   **`models/vqvae.py`:** Combines the encoder, decoder, and quantizer to form the complete VQVAE model.
*   **`models/classifier.py`:** Defines the architecture of the classification head.
*   **`models/vqvaecls.py`:**  Integrates the pre-trained, frozen VQVAE encoder with the classifier head for end-to-end classification.
*   **`train_vqvae.py`:** Script for training the VQVAE.
*   **`train_cls.py`:** Script for training the classifier using the frozen VQVAE encoder.
*   **`eval_cls.py`:** Script for evaluating the trained classifier.
*   **`infer_vqvae.py`:** Script for performing inference with the trained VQVAE (e.g., reconstructing images).
*   **`infer_cls.py`:** Script for performing inference with the trained classifier.

This modular design allows for easy modification and experimentation with different encoder/decoder architectures, quantization methods, and classifier heads.
