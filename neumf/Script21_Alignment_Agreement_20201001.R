# detach("package:SeuratWrappers", unload=TRUE)
# remove.packages("Seurat")
# library("multtest")
# # https://satijalab.org/seurat/install.html
# source("https://z.umn.edu/archived-seurat")
# library("Seurat")
# # packageVersion("Seurat")
# install.packages('Seurat')
library(Seurat)
packageVersion("Seurat")

# library(SeuratData)
library(liger)
library(Matrix) 
library(patchwork)
library(FNN)
library(dplyr)
library(ica)
library(cowplot)

source("/Users/meilanchen/Downloads/Meilan/Code/Alignment_Agreement_Function.R")
# liger::quantileAlignSNF		Quantile align (normalize) factor loadings
# SeuratWrappers::RunQuantileAlignSNF		Run quantileAlignSNF on a Seurat object




# Load the PBMC dataset
pbmc.10x <- read.table('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/pbmc_10X.expressionMatrix.txt',sep="\t",stringsAsFactors=F,header=T,row.names = 1)
pbmc.seqwell <- read.table('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/pbmc_SeqWell.expressionMatrix.txt',sep="\t",stringsAsFactors=F,header=T,row.names = 1)
pbmc.data = list(tenx=pbmc.10x, seqwell=pbmc.seqwell)

# Create liger object
liger.pbmc <- createLiger(pbmc.data)
liger.pbmc <- normalize(liger.pbmc)
# Can pass different var.thresh values to each dataset if one seems to be contributing significantly
# more genes than the other
liger.pbmc <- selectGenes(liger.pbmc, var.thresh = c(0.3, 0.875), do.plot = F)

# In this case, we have a precomputed set of variable genes
s.var.genes <- readRDS('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/var_genes.RDS')
liger.pbmc@var.genes <- s.var.genes
liger.pbmc <- scaleNotCenter(liger.pbmc)
# running suggestK on multiple cores can greatly decrease the runtime
# k.suggest <- suggestK(liger.pbmc, num.cores = 5, gen.new = T, return.results = T, plot.log2 = F,
#                       nrep = 5)

# Take the lowest objective of three factorizations with different initializations
# Multiple restarts are recommended for initial analyses since iNMF is non-deterministic
liger.pbmc <- optimizeALS(liger.pbmc, k=12, thresh = 5e-5, nrep = 3)
liger.pbmc <- runTSNE(liger.pbmc, use.raw = T)
p1 <- plotByDatasetAndCluster(liger.pbmc, return.plots = T)
# Plot by dataset
print(p1[[1]])
liger.pbmc <- quantileAlignSNF(liger.pbmc, resolution = 1.4, small.clust.thresh = 0)

liger.pbmc <- runTSNE(liger.pbmc)
p_a <- plotByDatasetAndCluster(liger.pbmc, return.plots = T) 
# Modify plot output slightly
p_a[[1]] <- p_a[[1]] + theme_classic() + theme(legend.position = c(0.85, 0.15)) + 
    guides(col=guide_legend(title = '', override.aes = list(size = 4)))

print(p_a[[1]])
print(p_a[[2]])




# load cluster labels from Seurat 10X PBMC analysis and original SeqWell publication
clusters_prior <- readRDS('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/tenx_seqwell_clusters.RDS')
tenx_c <- droplevels(clusters_prior[rownames(liger.pbmc@H[["tenx"]])])
seqwell_c <- droplevels(clusters_prior[rownames(liger.pbmc@H[["seqwell"]])])

# Set specific node order for cleaner visualization (can also leave to default value for 
# automatic ordering)
set_node_order = list(c(2, 6, 3, 4, 8, 1, 5, 7), c(1, 6, 2, 3, 4, 5, 7, 8, 9), c(5, 2, 3, 6, 1, 4))
# makeRiverplot(liger.pbmc, tenx_c, seqwell_c, min.frac = 0.05, node.order = set_node_order,
#               river.usr = c(0, 1, -0.6, 1.6))
makeRiverplot(liger.pbmc, tenx_c, seqwell_c, min.frac = 0.05)
png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_celltype.png", units="in", width=8, height=7, res=300)
makeRiverplot(liger.pbmc, tenx_c, seqwell_c, min.frac = 0.05)
dev.off()
# calculate liger alignment

liger.result <- data.frame(liger.pbmc@ tsne.coords,
                           cluster = liger.pbmc@clusters, 
                           celltype =c(levels(tenx_c)[tenx_c], levels(seqwell_c)[seqwell_c]), 
                           Data = c(rep("tenx", length(tenx_c)), rep("seqwell", length(seqwell_c))))
colnames(liger.result) <- c("tSNE_1", "tSNE_2", "cluster", "celltype", "dataset")
library(tidyverse)
p7 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = dataset))+
    geom_point(size=0.5)+
    ggtitle("LIGER")+
    theme_classic()

p8 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = celltype))+
    geom_point(size=0.5)+
    ggtitle("LIGER")+
    theme_classic()
p8

p19 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = cluster))+
    geom_point(size=0.5)+
    ggtitle("LIGER")+
    theme_classic()
p19

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_liger_cluster.png", units="in", width=5, height=5, res=300)
p19
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_liger_data.png", units="in", width=5, height=5, res=300)
p7
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_liger_celltype.png", units="in", width=5, height=5, res=300)
p8
dev.off()



object <- liger.pbmc
ndims = k =22
dr = object@H.norm
dr.original = list(object@H[[1]], object@H[[2]])

liger_alignment <- alignment(dr = dr, dr.original = dr.original, k=k, rand.seed=1324)
liger_alignment
# [1] 0.6593178



liger.agreement <- agreement(dr = dr.2, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324)
liger.agreement
# [1] 0.2125465

##################################################################################
## Method: dp
##################################################################################
library(dirichletprocess)

dp.pbmc <- createLiger(pbmc.data)
dp.pbmc <- normalize(dp.pbmc)
# Can pass different var.thresh values to each dataset if one seems to be contributing significantly
# more genes than the other
dp.pbmc <- selectGenes(dp.pbmc, var.thresh = c(0.3, 0.875), do.plot = F)


dp.pbmc <- scaleNotCenter(dp.pbmc)
dp.pbmc <- optimizeALS(dp.pbmc, k=4, thresh = 5e-5, nrep = 3)
dp.pbmc <- runTSNE(dp.pbmc, use.raw = T)

dp.pbmc <- quantileAlignSNF(dp.pbmc, resolution = 1.4, small.clust.thresh = 0)
dp.p1 <- plotByDatasetAndCluster(dp.pbmc, return.plots = T)
# Plot by dataset
print(dp.p1[[1]])
print(dp.p1[[2]])

nmf <- dp.pbmc@ H.norm
scale.nmf <- scale(nmf)
dp <-  DirichletProcessMvnormal(scale.nmf, alphaPriors = c(2, 4))
dp <- Fit(dp, 1000)


dp.result <- data.frame(dp.pbmc@ tsne.coords,cluster = as.factor(dp$clusterLabels), 
                        celltype =c(levels(tenx_c)[tenx_c], levels(seqwell_c)[seqwell_c]), 
                        Data = c(rep("tenx", length(tenx_c)), rep("seqwell", length(seqwell_c))))

colnames(dp.result) <- c("tSNE_1", "tSNE_2", "cluster", "celltype", "dataset")
p9 <- ggplot(data = dp.result, aes(x = tSNE_1, y = tSNE_2, color = dataset))+
    geom_point(size=0.5)+
    ggtitle("Dirichlet Process")+
    theme_classic()

p10 <- ggplot(data = dp.result, aes(x = tSNE_1, y = tSNE_2,color = celltype))+
    geom_point(size=0.5)+
    ggtitle("Dirichlet Process")+
    theme_classic()
p10

p10.2 <- ggplot(data = dp.result, aes(x = tSNE_1, y = tSNE_2,color = cluster))+
    geom_point(size=0.5)+
    ggtitle("Dirichlet Process")+
    theme_classic()
p10.2

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_dp_data.png", units="in", width=5, height=5, res=300)
p9
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_dp_celltype.png", units="in", width=5, height=5, res=300)
p10
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_dp_cluster.png", units="in", width=5, height=5, res=300)
p10.2
dev.off()

# ##################################################################
# CCA
# ##################################################################
remove.packages("Seurat")
library("multtest")
# # https://satijalab.org/seurat/install.html
source("https://z.umn.edu/archived-seurat")

library("Seurat")
packageVersion("Seurat")
library(FNN)
library(dplyr)
library(cowplot)
library(ica)
library(irlba)


cca.seqwell <- CreateSeuratObject(raw.data =pbmc.seqwell,
                                  project = "seqwell")
cca.seqwell@meta.data[, "data"] <- rep("seqwell", nrow(cca.seqwell@meta.data))
cca.seqwell <- NormalizeData(object = cca.seqwell)
cca.seqwell <- ScaleData(object = cca.seqwell)
cca.seqwell <- FindVariableGenes(object = cca.seqwell, do.plot = FALSE)


cca.10x  <- CreateSeuratObject(raw.data =pbmc.10x,
                               project = "tenx")
cca.10x@meta.data[, "data"] <- rep("tenx", nrow(cca.10x@meta.data))
cca.10x <- NormalizeData(object = cca.10x)
cca.10x <- ScaleData(object = cca.10x)
cca.10x <- FindVariableGenes(object = cca.10x, do.plot = FALSE)

hvg.10x <- rownames(x = head(x = cca.10x@hvg.info, n = 2000))
hvg.seqwell <- rownames(x = head(x = cca.seqwell@hvg.info, n = 2000))
hvg.union <- union(x = hvg.seqwell, y = hvg.10x)

cca.pbmc <- RunCCA(object = cca.seqwell, object2 = cca.10x, genes.use = hvg.union, num.cc = 22)


DimPlot(object = cca.pbmc , reduction.use = "cca", group.by = "data", pt.size = 0.5, 
        do.return = TRUE)
cca.pbmc <- AlignSubspace(object = cca.pbmc, reduction.type = "cca", grouping.var = "data", 
                          dims.align = 1:13)
cca.pbmc <- RunTSNE(object = cca.pbmc, reduction.use = "cca.aligned", dims.use = 1:13, 
                    do.fast = TRUE)

cca.pbmc <- FindClusters(object = cca.pbmc, reduction.type = "cca.aligned", dims.use = 1:13, 
                         save.SNN = TRUE)

# find markers for every cluster compared to all remaining cells, report only the positive ones
cca.markers <- FindAllMarkers(cca.pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
cca.markers %>% group_by(cluster) %>% top_n(n = 2, wt = avg_logFC)
top10 <- cca.markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
cca.heatmap <- DoHeatmap(cca.pbmc, genes = top10$gene) 
png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_cca_heatmap.png", units="in", width=25, height=15, res=300)
cca.heatmap 
dev.off()

cca.result <- data.frame(cca.pbmc@dr$tsne@cell.embeddings, cluster = cca.pbmc@meta.data$res.0.8, 
                         celltype =c(levels(tenx_c)[tenx_c], levels(seqwell_c)[seqwell_c]), 
                         Data = c(rep("tenx", length(tenx_c)), rep("seqwell", length(seqwell_c))))
colnames(liger.result) <- c("tSNE_1", "tSNE_2", "cluster", "celltype", "dataset")
library(tidyverse)
p1.1 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = dataset))+
    geom_point(size=0.5)+
    ggtitle("CCA")+
    theme_classic()

p1.2 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = celltype))+
    geom_point(size=0.5)+
    ggtitle("CCA")+
    theme_classic()

p1.3 <- ggplot(data = liger.result, aes(x = tSNE_1, y = tSNE_2,color = cluster))+
    geom_point(size=0.5)+
    ggtitle("CCA")+
    theme_classic()



png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_cca_cluster.png", units="in", width=5, height=5, res=300)
p1.3
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_cca_data.png", units="in", width=5, height=5, res=300)
p1.1
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_cca_celltype.png", units="in", width=5, height=5, res=300)
p1.2
dev.off()

p1 <- TSNEPlot(object = cca.pbmc, group.by = "data", do.return = TRUE, pt.size = 0.5)
p2 <- TSNEPlot(object = cca.pbmc, do.return = TRUE, pt.size = 0.5)
p1
# plot_grid(p1, p2)

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc_cca.png", units="in", width=5, height=5, res=300)
p1
dev.off()

# calculate cca alignment

object <- cca.pbmc
ndims = k = 22
dr <- (object@dr$cca.aligned@cell.embeddings)
dr.original.1 <- object@meta.data %>% 
    filter(data=="seqwell") %>% 
    rownames()
dr.original.2 <- object@meta.data %>% 
    filter(data=="tenx") %>% 
    rownames()

dr.original <- list(dr[dr.original.1,], dr[dr.original.2,])
cca_alignment <- alignment(dr=dr, dr.original =dr.original , k=22)
cca_alignment
# [1] 0.6114998

# calculate cca agreement
dr.list <- list(
    t(object@scale.data[object@var.genes, dr.original.1]),
    t(object@scale.data[object@var.genes, dr.original.2 ]))

dr2 <- lapply(dr.list, function(x) {
    icafast(x , nc =  ndims)$S
})

cca.agreement <- agreement(dr=dr2, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324)
cca.agreement
# [1] 0.08212284







# ################################################################
# MNN

remove.packages("Seurat")
# library("multtest")
# # https://satijalab.org/seurat/install.html
# source("https://z.umn.edu/archived-seurat")
# library("Seurat")
# packageVersion("Seurat")
install.packages('Seurat')
library(Seurat)
library(SeuratWrappers)
library(scran)
# Recall that groups for marker detection
# are automatically defined from 'colLabels()'. 
var.genes <- readRDS('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/var_genes.RDS')
pbmc.10x.re <- pbmc.10x[var.genes,]
pbmc.seqwell.re <- pbmc.seqwell[var.genes,]
mnn.pbmc.data <- cbind(pbmc.10x.re, pbmc.seqwell.re)
mnn.pbmc <- CreateSeuratObject(mnn.pbmc.data,
                               project = "mnn.pbmc")
mnn.pbmc@meta.data$data <- c(rep("tenx",ncol(pbmc.10x)), rep("seqwell",ncol(pbmc.seqwell)))


mnn.pbmc <- NormalizeData(mnn.pbmc)
mnn.pbmc <- ScaleData(mnn.pbmc)
mnn.pbmc <- FindVariableFeatures(object = mnn.pbmc,
                              x.low.cutoff = 0.0125, x.high.cutoff = 3,
                              y.cutoff = 0.5,
                              y.high.cutoff = Inf)

mnn.pbmc <- RunFastMNN(object.list = SplitObject(mnn.pbmc, split.by = "data"))

mnn.pbmc <- FindNeighbors(mnn.pbmc, reduction = "mnn", k.param = 22, dims = 1:30)
mnn.pbmc <- FindClusters(mnn.pbmc, n.iter = 100, resolution = 1)

# head(panda@meta.data)
mnn.pbmc@meta.data$seurat_clusters


mnn.pbmc <- RunTSNE(object = mnn.pbmc,
                 perplexity = 20,
                 reduction = "mnn",
                 dims.use = 1:2,
                 do.fast = TRUE)

mnn.result <- data.frame(Embeddings(mnn.pbmc[["tsne"]]), cluster = mnn.pbmc@meta.data$seurat_clusters, 
                           celltype =c(levels(tenx_c)[tenx_c], levels(seqwell_c)[seqwell_c]), 
                           Data = c(rep("tenx", length(tenx_c)), rep("seqwell", length(seqwell_c))))
colnames(mnn.result) <- c("tSNE_1", "tSNE_2", "cluster", "celltype", "dataset")
library(tidyverse)
p2 <- ggplot(data = mnn.result, aes(x = tSNE_1, y = tSNE_2,color = dataset))+
    geom_point(size=0.5)+
    ggtitle("MNN")+
    theme_classic()

p3 <- ggplot(data = mnn.result, aes(x = tSNE_1, y = tSNE_2,color = celltype))+
    geom_point(size=0.5)+
    ggtitle("MNN")+
    theme_classic()

p3.2 <- ggplot(data = mnn.result, aes(x = tSNE_1, y = tSNE_2,color = cluster))+
    geom_point(size=0.5)+
    ggtitle("MNN")+
    theme_classic()



png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_mnn_cluster.png", units="in", width=5, height=5, res=300)
p3.2
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_mnn_data.png", units="in", width=5, height=5, res=300)
p2
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_mnn_celltype.png", units="in", width=5, height=5, res=300)
p3
dev.off()

library(ComplexHeatmap )
# find markers for every cluster compared to all remaining cells, report only the positive ones
mnn.pbmc.markers <- FindAllMarkers(mnn.pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
mnn.pbmc.markers %>% group_by(cluster) %>% top_n(n = 2, wt = avg_logFC)
top10 <- mnn.pbmc.markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
DoHeatmap(mnn.pbmc, features = top10$gene) + NoLegend()
mnn.pbmc.markers2 <- mnn.pbmc.data[ top10$gene,]
dim(mnn.pbmc.markers2)
mnn.pbmc.markers2[1:20, 1:5]
Heatmap(mnn.pbmc.markers2)


mnn.heatmap <- heatmap(as.matrix(mnn.pbmc.markers2))
png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_mnn_heatmap.png", units="in", width=5, height=5, res=300)
heatmap(as.matrix(mnn.pbmc.markers2))
dev.off()

object <- mnn.pbmc
ndims = k = 22
dr <- (object@reductions$mnn) @ cell.embeddings 
str(dr)
dr.original.2 <- object@meta.data %>% 
    filter(data=="seqwell") %>% 
    rownames()
dr.original.1 <- object@meta.data %>% 
    filter(data=="tenx") %>% 
    rownames()

dr.original <- list(dr[dr.original.1,], dr[dr.original.2,])
mnn_alignment <- alignment(dr=dr, dr.original =dr.original , k=22)
mnn_alignment
# [1] 0.5279654

# calculate agreement

dr.list <- list(
    t(object@assays$RNA[,dr.original.1]),
    t(object@assays$RNA[, dr.original.2 ]))
library(irlba)

# compute PCA
dr2 <- lapply(dr.list, function(x) {
    prcomp_irlba(t(x),n = ndims,scale. = (colSums(x) > 0), center = F
    )$rotation
})


mnn.agreement <- agreement(dr=dr2, dr.original = dr.original ,ndims =ndims, k=k, rand.seed=1324)
mnn.agreement
# 1] 0.1558108




# ################################################################
# Harmony


library(harmony)
library(data.table)

var.genes <- readRDS('/Users/meilanchen/Downloads/Meilan/Data/pbmc_alignment/var_genes.RDS')
pbmc.10x.re <- pbmc.10x[var.genes,]
pbmc.seqwell.re <- pbmc.seqwell[var.genes,]
harmony.pbmc.data <- cbind(pbmc.10x.re, pbmc.seqwell.re)
harmony.pbmc <- CreateSeuratObject(harmony.pbmc.data,
                               project = "harmony.pbmc")
harmony.pbmc@meta.data$data <- c(rep("tenx",ncol(pbmc.10x)), rep("seqwell",ncol(pbmc.seqwell)))


harmony.pbmc <- NormalizeData(harmony.pbmc)
harmony.pbmc <- ScaleData(harmony.pbmc)
harmony.pbmc <- FindVariableFeatures(object = harmony.pbmc,
                                 x.low.cutoff = 0.0125, x.high.cutoff = 3,
                                 y.cutoff = 0.5,
                                 y.high.cutoff = Inf)

harmony.pbmc <- RunPCA(object = harmony.pbmc,
                npcs = 50,
                approx = FALSE,
                verbose = FALSE)
harmony.pbmc <- RunHarmony(harmony.pbmc, group.by.vars = "data", sigma=0.2,
                    lambda = 5, 
                    block.size = 0.01,
                    nclust =3, plot_convergence = T)

harmony.pbmc <- FindNeighbors(harmony.pbmc, reduction = "harmony", k.param = 22, dims = 1:50)
harmony.pbmc <- FindClusters(harmony.pbmc, n.iter = 100, resolution = 1)

# head(panda@meta.data)
harmony.pbmc@meta.data$seurat_clusters


harmony.pbmc <- RunTSNE(object = harmony.pbmc,
                    perplexity = 20,
                    reduction = "harmony",
                    dims.use = 1:2,
                    do.fast = TRUE)

harmony.result <- data.frame(Embeddings(harmony.pbmc[["tsne"]]), 
                         cluster = harmony.pbmc@meta.data$seurat_clusters,  
                         celltype =c(levels(tenx_c)[tenx_c], levels(seqwell_c)[seqwell_c]), 
                         Data = c(rep("tenx", length(tenx_c)), rep("seqwell", length(seqwell_c))))
colnames(harmony.result) <- c("tSNE_1", "tSNE_2", "cluster", "celltype", "dataset")
library(tidyverse)
p4 <- ggplot(data = harmony.result, aes(x = tSNE_1, y = tSNE_2,color = dataset))+
    geom_point(size=0.5)+
    ggtitle("Harmony")+
    theme_classic()

p5 <- ggplot(data = harmony.result, aes(x = tSNE_1, y = tSNE_2,color = celltype))+
    geom_point(size=0.5)+
    ggtitle("Harmony")+
    theme_classic()

p5.2 <- ggplot(data = harmony.result, aes(x = tSNE_1, y = tSNE_2,color = cluster))+
    geom_point(size=0.5)+
    ggtitle("Harmony")+
    theme_classic()



png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_harmony_cluster.png", units="in", width=5, height=5, res=300)
p5.2
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_harmony_data.png", units="in", width=5, height=5, res=300)
p4
dev.off()

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc/pbmc_harmony_celltype.png", units="in", width=5, height=5, res=300)
p5
dev.off()







object <- harmony.pbmc
ndims = k = 22
dr <- (object@reductions$harmony) @ cell.embeddings 

dr.original.2 <- object@meta.data %>% 
    filter(data=="seqwell") %>% 
    rownames()
dr.original.1 <- object@meta.data %>% 
    filter(data=="tenx") %>% 
    rownames()

dr.original <- list(dr[dr.original.1,], dr[dr.original.2,])
harmony_alignment <- alignment(dr=dr, dr.original =dr.original , k=22)
harmony_alignment
# [1] 0.6622269

# calculate agreement

dr.list <- list(
    (object@assays$RNA[,dr.original.1]),
    (object@assays$RNA[, dr.original.2 ]))
library(irlba)

# compute PCA
dr2 <- lapply(dr.list, function(x) {
    prcomp_irlba((x),n = ndims,scale. = (colSums(x) > 0), center = F
    )$rotation
})


harmony.agreement <- agreement(dr=dr2, dr.original = dr.original, ndims =ndims, k=k, rand.seed=1324)
harmony.agreement
# [1] 0.1189382


library(ggplot2)
alignment.result <- data.frame(Value = c(cca_alignment, liger_alignment, mnn_alignment, harmony_alignment), 
                               Method = c("CCA", "Liger", "MNN", "Harmony"))
bar1 <-ggplot(data=alignment.result , aes(x=Method, y=Value, fill=Method)) +
    geom_bar(stat="identity")+
    labs(title="Alignment", x ="", y = "")+
    geom_text(aes(label=round(Value,2)), vjust=-0.3, size=3.5)+
    theme_minimal()
bar1

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc_alignment.png", units="in", width=5, height=5, res=300)
bar1
dev.off()


agreement.result <- data.frame(Value = c(cca.agreement, liger.agreement, mnn.agreement, harmony.agreement), 
                               Method = c("CCA", "Liger", "MNN", "Harmony"))
bar2 <-ggplot(data=agreement.result, aes(x=Method, y=Value, fill=Method)) +
    geom_bar(stat="identity")+
    labs(title="Agreement", x ="", y = "")+
    geom_text(aes(label=round(Value,2)), vjust=-0.3, size=3.5)+
    theme_minimal()
bar2

png(filename="/Users/meilanchen/Downloads/Meilan/Figures/pbmc_agreement.png", units="in", width=5, height=5, res=300)
bar2
dev.off()