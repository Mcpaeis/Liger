

library(FNN)
library(Matrix) 

alignment <- function(dr = dr, dr.original = dr.original, k=10, rand.seed=1324){
    
    cells.use <- rownames(dr)
    num_cells <- length(c(cells.use))
    num_drs <- ncol(dr)
    N <- length(dr.original)
    set.seed(rand.seed)
    min_cells <- min(sapply(dr.original, function(x) {
        nrow(x)
    }))
    
    sampled_cells <- unlist(lapply(1:N, function(x) {
        sample(rownames(dr.original[[x]]), min_cells)
    }))
    
    
    knn_graph <- get.knn(dr[sampled_cells,], k) 
    dataset <- unlist(sapply(1:N, function(x) {
        rep(paste0('group', x), nrow(dr.original[[x]]))
    }))
    
    names(dataset) <- rownames(dr)
    dataset <- dataset[sampled_cells]
    num_sampled <- N * min_cells
    num_same_dataset <- rep(k, num_sampled)
    
    alignment_per_cell <- c()
    
    for (i in 1:num_sampled) {
        inds <- knn_graph$nn.index[i, ]
        num_same_dataset[i] <- sum(dataset[inds] == dataset[i])
        alignment_per_cell[i] <- 1 - (num_same_dataset[i] - (k / N)) / (k - k / N)
    }
    
    return(mean(alignment_per_cell))
    
    
}


# helper function of nmf_hals
nonneg <- function(x, eps = 1e-16) {
    x[x < eps] <- eps
    return(x)
}

frobenius_prod <- function(X, Y) {
    sum(X * Y)
}
# Hierarchical alternating least squares for regular NMF
nmf_hals <- function(A, k, max_iters = 500, thresh = 1e-4, reps = 20, W0 = NULL, H0 = NULL) {
    m <- nrow(A)
    n <- ncol(A)
    if (is.null(W0)) {
        W0 <- matrix(abs(runif(m * k, min = 0, max = 2)), m, k)
    }
    if (is.null(H0)) {
        H0 <- matrix(abs(runif(n * k, min = 0, max = 2)), n, k)
    }
    W <- W0
    rownames(W) <- rownames(A)
    H <- H0
    rownames(H) <- colnames(A)
    
    # alpha = frobenius_prod(A %*% H, W)/frobenius_prod(t(W)%*%W,t(H)%*%H)
    # W = alpha*W
    
    for (i in 1:k)
    {
        W[, i] <- W[, i] / sum(W[, i]^2)
    }
    
    delta <- 1
    iters <- 0
    # pb <- txtProgressBar(min=0,max=max_iters,style=3)
    iter_times <- rep(0, length(max_iters))
    objs <- rep(0, length(max_iters))
    obj <- norm(A - W %*% t(H), "F")^2
    # print(obj)
    while (delta > thresh & iters < max_iters) {
        start_time <- Sys.time()
        obj0 <- obj
        HtH <- t(H) %*% H
        AH <- A %*% H
        for (i in 1:k)
        {
            W[, i] <- nonneg(W[, i] + (AH[, i] - (W %*% HtH[, i])) / (HtH[i, i]))
            W[, i] <- W[, i] / sqrt(sum(W[, i]^2))
        }
        
        AtW <- t(A) %*% W
        WtW <- t(W) %*% W
        for (i in 1:k)
        {
            H[, i] <- nonneg(H[, i] + (AtW[, i] - (H %*% WtW[, i])) / (WtW[i, i]))
        }
        
        obj <- norm(A - W %*% t(H), "F")^2
        # print(obj)
        delta <- abs(obj0 - obj) / (mean(obj0, obj))
        iters <- iters + 1
        end_time <- Sys.time()
        iter_times[iters] <- end_time - start_time
        objs[iters] <- obj
        # setTxtProgressBar(pb,iters)
    }
    cat("\nConverged in", end_time - start_time, "seconds,", iters, "iterations. Objective:", obj, "\n")
    # boxplot(iter_times)
    
    return(list(W, H, cumsum(iter_times), objs))
}

# liger::quantileAlignSNF		Quantile align (normalize) factor loadings
# SeuratWrappers::RunQuantileAlignSNF		Run quantileAlignSNF on a Seurat object
library(ica)
library(irlba)





agreement <- function(dr=dr, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324){
    
    set.seed(rand.seed)
    ns <- sapply(dr, nrow)
    n <- sum(ns)
    jaccard_inds <- c()
    distorts <- c()
    original <- dr.original 
    for (i in 1:length(dr)) {
        
        jaccard_inds_i <- c()
        original.i <- original[[i]]
        fnn.1 <- get.knn(dr[[i]], k = k)
        fnn.2 <- get.knn(original.i, k = k)
        jaccard_inds_i <- c(jaccard_inds_i, sapply(1:ns[i], function(i) {
            intersect <- intersect(fnn.1$nn.index[i, ], fnn.2$nn.index[i, ])
            union <- union(fnn.1$nn.index[i, ], fnn.2$nn.index[i, ])
            length(intersect) / length(union)
        }))
        jaccard_inds_i <- jaccard_inds_i[is.finite(jaccard_inds_i)]
        jaccard_inds <- c(jaccard_inds, jaccard_inds_i)
        
        distorts <- c(distorts, mean(jaccard_inds_i))
    }
    
    agreement <- mean(jaccard_inds)
    return(agreement)
    
}



# Example
# calculate liger alignment agreement
# object <- liger.pbmc
# ndims = k =22
# dr = object@H.norm
# dr.original = list(object@H[[1]], object@H[[2]])
# 
# liger_alignment <- alignment(dr = dr, dr.original = dr.original, k=k, rand.seed=1324)
# liger_alignment
# # [1] 0.6593178
# 
# 
# 
# liger.agreement <- agreement(dr = dr.2, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324)
# liger.agreement

# calculate cca alignment agreement

# object <- cca.pbmc
# ndims = k = 22
# dr <- (object@dr$cca.aligned@cell.embeddings)
# dr.original.1 <- object@meta.data %>% 
#     filter(data=="seqwell") %>% 
#     rownames()
# dr.original.2 <- object@meta.data %>% 
#     filter(data=="tenx") %>% 
#     rownames()
# 
# dr.original <- list(dr[dr.original.1,], dr[dr.original.2,])
# cca_alignment <- alignment(dr=dr, dr.original =dr.original , k=22)
# cca_alignment
# # [1] 0.6114998
# 
# # calculate cca agreement
# dr.list <- list(
#     t(object@scale.data[object@var.genes, dr.original.1]),
#     t(object@scale.data[object@var.genes, dr.original.2 ]))
# 
# dr2 <- lapply(dr.list, function(x) {
#     icafast(x , nc =  ndims)$S
# })
# 
# cca.agreement <- agreement(dr=dr2, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324)
# cca.agreement
