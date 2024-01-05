# Liger
Unsupervised Deep Learning Procedure for Batch Effect Correction in single-cell RNA-Seq data using Neural Matrix Factorization (NeuMF), an adaptation of Non-negative matrix factorization. 

![Screen Shot 2021-01-22 at 7 03 23 PM](https://github.com/Mcpaeis/Liger/assets/26014460/89bc42db-ef8f-452d-b6d5-ff81444dbd3c)


### The implementation process is described below.

## (A) Neural Matrix Factorizarion Steps -  NeuMF Paper

1.   Usually given an "object 1 by object 2" observation matrix M with explicit feedback data.

---

2.   Two learning approaches can be adopted: 

*   Pointwise learning: follows a regression framework by minimizing the squred loss between *yPredicted* and *yObserved*.
*   Pairwise learning: this ranks observed entries higher relative to unobserved entries, and maximizes the margin between them.

---

3. Electing pointwise learning, the training phase involves searching for the optimal parameters that minimizes this squared loss over a reduced (k-dimensional) feature space.

---

*   Predictions can then be made for the unobserved entries.
*   In essence, both the observed and unobserved entries have been approximated by a non-linear function, which presumably improves upon linear(dot product) approximation.

---

What's often done(and adopted by the paper) is to restructure the problem to make use of implicit data (more easy to collect compared to explicit data)

So in this case, the observation matrix M is converted into a binary interaction matrix P. The training prediction processes are similar as in the case of learning with M.

---
---

## (B) Recovering the gene expression matrix(normalized) from LIGER ##

1. Pass the downsampled control and interferon-stimulated PBMCs to the LIGER function.

2. The LIGER function returns a normalized **H1, H2**, **V1, V2**, and **W**(shared) by performing non-negative matrix factorization.

3. These matrices can be used to recover a dense representation of the original expression matrices.

4. ** The "cell paper" then achieves integrative clustering by buildng a shared neighbourhood graph following the five steps of "Joint clustering and factor normalization" under the STAR methods.

---
---

## (C) TensorFlow Implementation Guidelines

### Get data:
1. A first option to getting the gene expression matrices is to recover them from the LIGER output.



  *   The recovered matrices are a dense representation of the original expression matrices
  *  Once the data is obtained this way, NeuMF cannot be applied for the simple reason that there are presumably no negative(unobserved/missing) instances.

2. A second option will be to get the raw representation. This option will allow the application of NeuMF.

---
### Implementation Steps:

1. Create a LIGER object with the raw data (stim & ctrl datasets)
  -  This allows the recovery of a sparse gene by cell matrix.
  - Cells not expressing any genes and genes not expressed in any cell are removed.
  - Any remaining 0 is thus a "true" missing expression.
  - Two sparse matrix representations will result; "ctr_sparse" and "stim_sparse"

2. Feed the two sparse representations separately through the network. 
  - This will produce two dense representations of the sparse versions.
  - The NeuMF will leverage any existing non-linear relationship between the cells and genes whiles maintaining the usual linear operations for better predictions.

3. Using these two dense representations create a LIGER object with placeholders for H1, H2, V1, V2 and the shared matrix W.

4. Scale and normalize the values of the resulting object. This ensures each gene has the same variance and also accounts for varying sequencing depths.

5. Run the NMF algorithm from LIGER to get values for the matrices in (3)

6. Build the shared neighbourhood graph and carry out the clustering. Implementation can be done with the LIGER package.

---
---

## (D) Why this gives better clusters

    ** will denote the paper implementation
    *** will denote the corresponding implemntation in NeuMF
---

1. ### Factorization Method:

---
  ** -- Finds the matrices (of reduced dimensions) in (C)(3) by minimizing the penealized frobenius norm squared error between the observed and estimated matrices.

  *** -- Finds some two lower dimensional matrics say U & V, that densely approximates the observed expression matrix by minimizing the MSE.

  * The two loss functions are similar and differ only by a transformation.

---

2. ### Optimization Algorithm:

---

  **  -- Uses Block Cordinate Descent. It iteratively minimizes the factorization objective by making use of profiling. Convergence (local min) is guaranteed.

  *** -- Uses Stochastic Gradient Descent. Simultaneously update model network parameters to minimze the loss function. Convergence is not guaranteed but saddle points in deep networks have shown to produce optimal functions than their shallow counterparts (BCD can be formulated as such).

---

3. ### Prediction Function:

---

  **  -- Basically a linear approximation.

  *** -- Introduces non-linearities via the activation fucntions.

---

* For any cell gene expression matrix, NeuMF will always perform atleast as well as NMF Liger implementation. Performance will be indistninguishable if only linear (unlikely) relationship is present.

* For the same factor specification k, performing Liger NMF on a dense output from NeuMF will not have any impact.

* The above is why it makes sense to create a LIGER object with the matrices from NeuMF and cluster based on LIGERs NNB implementation.

---
