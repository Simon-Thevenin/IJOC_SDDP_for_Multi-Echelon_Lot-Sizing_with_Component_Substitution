import pstats
p = pstats.Stats('profiler.txt')
p.sort_stats('cumulative').print_stats(100000)