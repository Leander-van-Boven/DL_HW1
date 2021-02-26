df <- read.csv('data.csv')

library(ggplot2)

df$bi[df$batch=='16'] <- 0
df$bi[df$batch=='32'] <- 1
df$bi[df$batch=='64'] <- 2
df$network[df$network=='efnb3'] <- 'EfficientNetB3'
df$network[df$network=='efnb4'] <- 'EfficientNetB4'

ggplot(df, mapping=aes(x=bi, y=acc, color=optimizer)) + 
  #geom_bar(stat="identity", position='dodge') + 
  geom_point(size=2) +
  geom_line(alpha=0.5, size=1) + 
  facet_wrap(~network) + 
  scale_x_discrete(name='Batch size', labels=c('16', '32', '64'), limits=c(0, 1, 2)) +
  scale_y_continuous(name='Accuracy', limits = c(0,0.8)) +
  theme_bw() +
  theme(legend.position='bottom', legend.title = element_blank(), legend.margin = margin())
    
ggplot(df, mapping=aes(x=bi, y=seconds, color=optimizer)) + 
  #geom_bar(stat="identity", position='dodge') + 
  geom_point(size=2) +
  geom_line(alpha=0.5, size=1) + 
  facet_wrap(~network) + 
  scale_x_discrete(name='Batch size', labels=c('16', '32', '64'), limits=c(0, 1, 2)) +
  scale_y_continuous(name='Time (s)') +
  theme_bw() +
  theme(legend.position='bottom', legend.title = element_blank(), legend.margin = margin())
