setwd("~/db/Classes/ML/Project/DREAM5-CNNs/")
auc_ap = read.table("oversample_auc_ap.txt",skip = 1)
even_index = seq(1,39,2)
me_auc_ap = auc_ap[even_index,]
me_auc_ap = cbind(read.table("data/train_TFs.txt")[,1],me_auc_ap)
colnames(me_auc_ap) = c("tf","train AUC","train AP","val AUC","val AP")
library(reshape2)
me_auc_ap = melt(me_auc_ap)

hk_auc_ap = auc_ap[-even_index,]
hk_auc_ap = cbind(read.table("data/train_TFs.txt")[,1], hk_auc_ap)
colnames(hk_auc_ap) = c("tf","train AUC","train AP","val AUC","val AP")
hk_auc_ap = melt(hk_auc_ap)

# all_data = rbind(me_auc_ap, hk_auc_ap)
# all_data = cbind(c(rep("ME",20),rep("HK",20)),all_data)
# colnames(all_data) = c("array_type","tf","train AUC","train AP","val AUC","val AP")
# all_data = melt(all_data)

library(ggplot2)
me_plot = ggplot(data=me_auc_ap, aes(x=tf,y=value, group=variable, colour=variable)) +
  geom_line() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("High binder predictions for ME arrays") +
  xlab("Transcription Factor")

hk_plot = ggplot(data=hk_auc_ap, aes(x=tf,y=value, group=variable, colour=variable)) +
  geom_line() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("High binder predictions for HK arrays") +
  xlab("Transcription Factor")



# ggplot(data=all_data, aes(x=tf,y=value, group=array_type, colour=variable)) +
#   geom_line() + 
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   ggtitle("High binder predictions for ME arrays")

require(gridExtra)
grid.arrange(me_plot, hk_plot,ncol=2)

