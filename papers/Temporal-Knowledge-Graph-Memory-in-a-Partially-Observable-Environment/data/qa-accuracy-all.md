# QA accuracy for all agents

## KG Agent

### train

#### Memory Size 0

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 1.000 | 0.000 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 2

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 2.880 | 0.204 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 4

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 2.280 | 0.733 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 8

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 4.240 | 0.916 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 16

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 5.680 | 0.412 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 32

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 10.160 | 0.512 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 64

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 18.080 | 0.688 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 128

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 28.920 | 1.318 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 256

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 41.400 | 1.403 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 512

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 42.960 | 0.697 | SPARQL | BFS     | Uniform |              0.000 |

### Test

#### Memory Size 0

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 0.000 | 0.000 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 2

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 3.200 | 0.780 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 4

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 2.840 | 0.731 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 8

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 3.680 | 0.627 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 16

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 6.000 | 1.051 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 32

|  mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| ----: | ----: | :----- | :------ | :------ | -----------------: |
| 9.200 | 1.131 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 64

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 16.080 | 0.515 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 128

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 26.560 | 1.878 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 256

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 41.040 | 0.889 | SPARQL | BFS     | Uniform |              0.000 |

#### Memory Size 512

|   mean |   std | QA     | Explore | Evict   | avg_duration_hours |
| -----: | ----: | :----- | :------ | :------ | -----------------: |
| 42.080 | 0.845 | SPARQL | BFS     | Uniform |              0.000 |

## TKG Agent

### Train

#### Memory Size 0

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 2.000 | 0.400 | MRU | MRA     | LRU   |              0.025 |
| 2.000 | 0.400 | MRU | MFU     | LRU   |              0.026 |
| 2.000 | 0.400 | MFU | MFU     | LFU   |              0.026 |
| 2.000 | 0.400 | MRA | MFU     | LFU   |              0.025 |
| 2.000 | 0.400 | MFU | MRA     | LRU   |              0.024 |
| 2.000 | 0.400 | MFU | MRU     | LRU   |              0.025 |
| 2.000 | 0.400 | MFU | MRU     | LFU   |              0.024 |
| 2.000 | 0.400 | MRA | MRA     | LFU   |              0.025 |
| 2.000 | 0.400 | MRA | MFU     | LRU   |              0.025 |
| 2.000 | 0.400 | MFU | MFU     | LRU   |              0.025 |
| 2.000 | 0.400 | MFU | MRA     | LFU   |              0.025 |
| 2.000 | 0.400 | MRU | MRU     | LRU   |              0.024 |
| 2.000 | 0.400 | MRU | MRU     | LFU   |              0.027 |
| 2.000 | 0.400 | MRU | MFU     | LFU   |              0.025 |
| 2.000 | 0.400 | MRA | MRU     | LFU   |              0.025 |
| 2.000 | 0.400 | MRA | MRU     | LRU   |              0.026 |
| 2.000 | 0.400 | MRU | MRA     | LFU   |              0.025 |
| 2.000 | 0.400 | MRA | MRA     | LRU   |              0.026 |
| 1.960 | 0.557 | MFU | MFU     | FIFO  |              0.028 |
| 1.960 | 0.557 | MRU | MRA     | FIFO  |              0.028 |
| 1.960 | 0.557 | MRA | MFU     | FIFO  |              0.028 |
| 1.960 | 0.557 | MFU | MRA     | FIFO  |              0.029 |
| 1.960 | 0.557 | MRA | MRU     | FIFO  |              0.027 |
| 1.960 | 0.557 | MRU | MRU     | FIFO  |              0.030 |
| 1.960 | 0.557 | MRU | MFU     | FIFO  |              0.028 |
| 1.960 | 0.557 | MFU | MRU     | FIFO  |              0.029 |
| 1.960 | 0.557 | MRA | MRA     | FIFO  |              0.029 |

#### Memory Size 2

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 3.040 | 0.941 | MRA | MRA     | LFU   |              0.026 |
| 3.000 | 0.657 | MRU | MRA     | LFU   |              0.026 |
| 3.000 | 0.657 | MFU | MRA     | LFU   |              0.025 |
| 2.720 | 0.826 | MFU | MRA     | FIFO  |              0.029 |
| 2.720 | 0.826 | MRU | MRA     | FIFO  |              0.028 |
| 2.320 | 0.755 | MRA | MRA     | FIFO  |              0.028 |
| 2.240 | 1.031 | MFU | MRU     | LFU   |              0.026 |
| 2.240 | 1.031 | MFU | MFU     | LFU   |              0.026 |
| 2.240 | 1.031 | MRU | MRU     | LFU   |              0.025 |
| 2.240 | 1.031 | MRU | MFU     | LFU   |              0.025 |
| 2.200 | 0.438 | MFU | MFU     | FIFO  |              0.031 |
| 2.200 | 0.438 | MRU | MFU     | FIFO  |              0.030 |
| 2.200 | 0.438 | MFU | MRU     | FIFO  |              0.030 |
| 2.200 | 0.438 | MRU | MRU     | FIFO  |              0.028 |
| 2.160 | 1.046 | MRA | MFU     | LRU   |              0.024 |
| 2.160 | 0.916 | MRA | MFU     | LFU   |              0.027 |
| 2.160 | 0.916 | MRA | MRU     | LFU   |              0.024 |
| 2.160 | 0.686 | MRA | MRU     | FIFO  |              0.029 |
| 2.160 | 0.686 | MRA | MFU     | FIFO  |              0.030 |
| 2.160 | 1.046 | MRA | MRU     | LRU   |              0.026 |
| 2.040 | 0.408 | MRU | MRA     | LRU   |              0.027 |
| 2.040 | 0.408 | MFU | MRA     | LRU   |              0.026 |
| 1.960 | 0.388 | MRA | MRA     | LRU   |              0.025 |
| 1.920 | 0.733 | MFU | MRU     | LRU   |              0.025 |
| 1.920 | 0.733 | MFU | MFU     | LRU   |              0.026 |
| 1.920 | 0.733 | MRU | MFU     | LRU   |              0.026 |
| 1.920 | 0.733 | MRU | MRU     | LRU   |              0.023 |

#### Memory Size 4

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 3.200 | 0.310 | MRA | MRA     | FIFO  |              0.030 |
| 3.040 | 1.477 | MFU | MRU     | LFU   |              0.026 |
| 3.040 | 1.477 | MFU | MFU     | LFU   |              0.026 |
| 3.040 | 1.477 | MRU | MFU     | LFU   |              0.027 |
| 3.040 | 1.477 | MRU | MRU     | LFU   |              0.026 |
| 3.040 | 1.382 | MRA | MRA     | LFU   |              0.024 |
| 2.920 | 0.688 | MRU | MRA     | LFU   |              0.026 |
| 2.920 | 0.688 | MFU | MRA     | LFU   |              0.026 |
| 2.880 | 0.412 | MFU | MRA     | FIFO  |              0.031 |
| 2.880 | 0.412 | MRU | MRA     | FIFO  |              0.030 |
| 2.840 | 1.347 | MRA | MRU     | LFU   |              0.027 |
| 2.840 | 1.347 | MRA | MFU     | LFU   |              0.025 |
| 2.640 | 0.388 | MRA | MFU     | LRU   |              0.027 |
| 2.640 | 0.697 | MRU | MRU     | FIFO  |              0.027 |
| 2.640 | 0.697 | MRU | MFU     | FIFO  |              0.030 |
| 2.640 | 0.697 | MFU | MFU     | FIFO  |              0.030 |
| 2.640 | 0.697 | MFU | MRU     | FIFO  |              0.029 |
| 2.640 | 0.388 | MRA | MRU     | LRU   |              0.023 |
| 2.600 | 0.769 | MRA | MRA     | LRU   |              0.023 |
| 2.560 | 0.320 | MFU | MRU     | LRU   |              0.024 |
| 2.560 | 0.320 | MFU | MFU     | LRU   |              0.028 |
| 2.560 | 0.320 | MRU | MRU     | LRU   |              0.026 |
| 2.560 | 0.320 | MRU | MFU     | LRU   |              0.024 |
| 2.480 | 0.483 | MFU | MRA     | LRU   |              0.026 |
| 2.480 | 0.483 | MRU | MRA     | LRU   |              0.026 |
| 2.280 | 0.349 | MRA | MRU     | FIFO  |              0.029 |
| 2.280 | 0.349 | MRA | MFU     | FIFO  |              0.028 |

#### Memory Size 8

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 5.080 | 1.921 | MFU | MRA     | LFU   |              0.028 |
| 5.080 | 1.921 | MRU | MRA     | LFU   |              0.027 |
| 5.040 | 1.299 | MRA | MRA     | LFU   |              0.027 |
| 4.680 | 1.853 | MRA | MRU     | LFU   |              0.027 |
| 4.400 | 2.139 | MRA | MFU     | LFU   |              0.022 |
| 4.360 | 1.323 | MFU | MRU     | LFU   |              0.028 |
| 4.320 | 1.342 | MRU | MRU     | LFU   |              0.028 |
| 4.120 | 1.157 | MFU | MFU     | LFU   |              0.027 |
| 4.080 | 1.163 | MRU | MFU     | LFU   |              0.021 |
| 4.000 | 0.955 | MRU | MRA     | LRU   |              0.027 |
| 4.000 | 0.955 | MFU | MRA     | LRU   |              0.027 |
| 3.720 | 0.271 | MRA | MRU     | FIFO  |              0.030 |
| 3.720 | 0.271 | MRA | MFU     | FIFO  |              0.030 |
| 3.560 | 0.774 | MRA | MRA     | LRU   |              0.027 |
| 3.400 | 0.748 | MRU | MRA     | FIFO  |              0.029 |
| 3.400 | 0.748 | MFU | MRA     | FIFO  |              0.032 |
| 3.360 | 0.975 | MRU | MFU     | LRU   |              0.028 |
| 3.360 | 0.975 | MFU | MFU     | LRU   |              0.028 |
| 3.360 | 0.975 | MFU | MRU     | LRU   |              0.028 |
| 3.360 | 0.862 | MRA | MFU     | LRU   |              0.027 |
| 3.360 | 0.862 | MRA | MRU     | LRU   |              0.027 |
| 3.360 | 0.975 | MRU | MRU     | LRU   |              0.025 |
| 3.320 | 0.515 | MFU | MFU     | FIFO  |              0.031 |
| 3.320 | 0.515 | MFU | MRU     | FIFO  |              0.029 |
| 3.320 | 0.515 | MRU | MFU     | FIFO  |              0.031 |
| 3.320 | 0.515 | MRU | MRU     | FIFO  |              0.030 |
| 3.000 | 0.551 | MRA | MRA     | FIFO  |              0.030 |

#### Memory Size 16

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 7.840 | 0.871 | MRA | MFU     | LFU   |              0.027 |
| 7.240 | 0.889 | MRU | MRA     | LFU   |              0.027 |
| 7.120 | 1.204 | MFU | MFU     | LFU   |              0.029 |
| 7.040 | 1.023 | MFU | MRA     | LFU   |              0.029 |
| 6.960 | 0.784 | MRU | MFU     | LFU   |              0.030 |
| 6.760 | 1.750 | MFU | MRU     | LFU   |              0.029 |
| 6.680 | 1.542 | MRU | MRU     | LFU   |              0.028 |
| 6.640 | 0.763 | MRA | MRA     | LFU   |              0.028 |
| 6.040 | 2.381 | MRA | MRU     | LFU   |              0.029 |
| 5.840 | 0.814 | MRA | MRU     | LRU   |              0.030 |
| 5.760 | 1.148 | MFU | MFU     | LRU   |              0.028 |
| 5.760 | 1.148 | MRU | MFU     | LRU   |              0.028 |
| 5.720 | 0.816 | MRU | MRA     | FIFO  |              0.033 |
| 5.680 | 0.676 | MRA | MFU     | LRU   |              0.030 |
| 5.480 | 0.601 | MRA | MRA     | LRU   |              0.030 |
| 5.440 | 0.637 | MFU | MRA     | FIFO  |              0.031 |
| 5.240 | 0.731 | MRA | MRA     | FIFO  |              0.030 |
| 5.240 | 1.176 | MRU | MRU     | LRU   |              0.030 |
| 5.160 | 1.235 | MFU | MRU     | LRU   |              0.028 |
| 5.000 | 0.996 | MRA | MRU     | FIFO  |              0.032 |
| 4.960 | 0.916 | MFU | MRU     | FIFO  |              0.031 |
| 4.920 | 0.786 | MRA | MFU     | FIFO  |              0.030 |
| 4.720 | 0.873 | MRU | MRU     | FIFO  |              0.032 |
| 4.280 | 0.640 | MRU | MFU     | FIFO  |              0.030 |
| 4.160 | 0.709 | MFU | MRA     | LRU   |              0.029 |
| 3.840 | 0.650 | MFU | MFU     | FIFO  |              0.033 |
| 3.720 | 0.845 | MRU | MRA     | LRU   |              0.029 |

#### Memory Size 32

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 12.840 | 1.155 | MRA | MFU     | LFU   |              0.031 |
| 11.760 | 1.411 | MFU | MRA     | LFU   |              0.034 |
| 11.160 | 1.286 | MRU | MRA     | LFU   |              0.032 |
| 11.000 | 1.351 | MFU | MFU     | LFU   |              0.033 |
| 10.960 | 1.286 | MRA | MRA     | LFU   |              0.031 |
| 10.840 | 1.305 | MRU | MFU     | LFU   |              0.034 |
| 10.320 | 2.750 | MRA | MRU     | LFU   |              0.033 |
| 10.320 | 2.164 | MRU | MRU     | LFU   |              0.033 |
| 10.000 | 2.780 | MFU | MRU     | LFU   |              0.030 |
|  9.640 | 0.924 | MRA | MRU     | LRU   |              0.031 |
|  9.440 | 1.176 | MFU | MFU     | LRU   |              0.033 |
|  9.040 | 0.794 | MRA | MRU     | FIFO  |              0.034 |
|  8.960 | 0.585 | MFU | MRA     | LRU   |              0.035 |
|  8.840 | 0.774 | MRU | MRA     | LRU   |              0.033 |
|  8.480 | 1.136 | MFU | MRU     | LRU   |              0.034 |
|  8.400 | 0.748 | MRA | MRA     | LRU   |              0.033 |
|  8.400 | 0.829 | MRU | MFU     | FIFO  |              0.035 |
|  8.280 | 1.354 | MRU | MFU     | LRU   |              0.032 |
|  8.240 | 0.731 | MFU | MFU     | FIFO  |              0.034 |
|  8.160 | 0.585 | MFU | MRU     | FIFO  |              0.033 |
|  8.120 | 0.627 | MRA | MFU     | FIFO  |              0.035 |
|  8.000 | 0.400 | MRU | MRU     | FIFO  |              0.034 |
|  7.960 | 0.852 | MFU | MRA     | FIFO  |              0.035 |
|  7.960 | 1.632 | MRA | MFU     | LRU   |              0.034 |
|  7.840 | 0.408 | MRU | MRA     | FIFO  |              0.035 |
|  7.600 | 0.769 | MRA | MRA     | FIFO  |              0.033 |
|  7.440 | 0.833 | MRU | MRU     | LRU   |              0.032 |

#### Memory Size 64

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 21.480 | 1.275 | MRU | MRA     | LFU   |              0.040 |
| 20.720 | 2.111 | MFU | MRA     | LFU   |              0.037 |
| 20.640 | 2.114 | MRU | MFU     | LFU   |              0.038 |
| 20.320 | 2.010 | MRU | MRU     | LFU   |              0.039 |
| 20.160 | 1.882 | MRA | MFU     | LFU   |              0.041 |
| 20.000 | 1.863 | MRA | MRU     | LFU   |              0.039 |
| 19.920 | 1.657 | MRA | MRA     | LFU   |              0.039 |
| 19.680 | 1.505 | MFU | MRU     | LFU   |              0.039 |
| 19.280 | 1.723 | MFU | MFU     | LFU   |              0.038 |
| 16.920 | 1.800 | MRU | MRU     | LRU   |              0.038 |
| 16.640 | 1.341 | MRA | MRU     | LRU   |              0.038 |
| 16.480 | 1.489 | MFU | MRA     | LRU   |              0.037 |
| 16.440 | 1.416 | MRU | MFU     | LRU   |              0.040 |
| 16.360 | 1.148 | MRA | MFU     | LRU   |              0.037 |
| 16.040 | 0.991 | MFU | MFU     | LRU   |              0.040 |
| 15.720 | 3.689 | MFU | MRU     | LRU   |              0.039 |
| 15.400 | 1.513 | MRA | MRA     | LRU   |              0.039 |
| 14.920 | 1.078 | MRU | MRA     | LRU   |              0.040 |
| 14.640 | 0.265 | MFU | MRA     | FIFO  |              0.042 |
| 14.160 | 1.577 | MRA | MRA     | FIFO  |              0.038 |
| 14.080 | 1.040 | MRA | MRU     | FIFO  |              0.037 |
| 14.040 | 1.804 | MRA | MFU     | FIFO  |              0.041 |
| 13.880 | 1.217 | MRU | MRA     | FIFO  |              0.042 |
| 13.600 | 2.012 | MRU | MRU     | FIFO  |              0.041 |
| 13.320 | 1.588 | MRU | MFU     | FIFO  |              0.041 |
| 13.080 | 1.457 | MFU | MFU     | FIFO  |              0.040 |
| 12.800 | 0.780 | MFU | MRU     | FIFO  |              0.040 |

#### Memory Size 128

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 32.040 | 1.843 | MRA | MRA     | LFU   |              0.046 |
| 31.960 | 1.422 | MFU | MRA     | LFU   |              0.046 |
| 31.720 | 2.134 | MRU | MRA     | LFU   |              0.050 |
| 31.400 | 1.914 | MRU | MRU     | LFU   |              0.050 |
| 31.200 | 1.757 | MFU | MFU     | LFU   |              0.049 |
| 31.120 | 2.058 | MRU | MFU     | LFU   |              0.050 |
| 31.000 | 1.673 | MFU | MRU     | LFU   |              0.049 |
| 30.000 | 0.972 | MRA | MRA     | LRU   |              0.049 |
| 29.760 | 2.344 | MRA | MFU     | LFU   |              0.050 |
| 29.760 | 1.607 | MRA | MRU     | LFU   |              0.046 |
| 29.480 | 1.500 | MRU | MRA     | LRU   |              0.050 |
| 29.320 | 1.093 | MFU | MRA     | LRU   |              0.046 |
| 29.240 | 1.646 | MRU | MFU     | LRU   |              0.048 |
| 29.120 | 1.552 | MFU | MFU     | LRU   |              0.045 |
| 29.040 | 3.352 | MRA | MRU     | LRU   |              0.042 |
| 28.960 | 1.341 | MRA | MFU     | LRU   |              0.049 |
| 28.720 | 2.141 | MFU | MRU     | LRU   |              0.050 |
| 28.600 | 1.702 | MRU | MRU     | LRU   |              0.050 |
| 25.040 | 1.148 | MRA | MFU     | FIFO  |              0.050 |
| 25.000 | 1.187 | MRU | MFU     | FIFO  |              0.048 |
| 24.880 | 1.662 | MRA | MRU     | FIFO  |              0.047 |
| 24.880 | 1.774 | MRU | MRA     | FIFO  |              0.042 |
| 24.520 | 2.348 | MFU | MFU     | FIFO  |              0.046 |
| 24.480 | 1.204 | MRU | MRU     | FIFO  |              0.044 |
| 23.720 | 2.389 | MFU | MRU     | FIFO  |              0.049 |
| 23.560 | 1.680 | MRA | MRA     | FIFO  |              0.045 |
| 23.520 | 0.449 | MFU | MRA     | FIFO  |              0.050 |

#### Memory Size 256

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 41.800 | 1.824 | MRU | MRU     | LRU   |              0.064 |
| 41.520 | 1.287 | MRA | MRU     | LFU   |              0.065 |
| 41.360 | 1.203 | MRU | MRU     | LFU   |              0.059 |
| 41.240 | 1.031 | MRA | MRU     | LRU   |              0.060 |
| 41.200 | 0.955 | MRU | MRA     | LFU   |              0.064 |
| 41.080 | 1.647 | MFU | MRA     | LFU   |              0.064 |
| 41.040 | 1.891 | MRA | MFU     | LRU   |              0.062 |
| 41.040 | 1.261 | MFU | MRU     | LRU   |              0.066 |
| 41.040 | 1.646 | MRA | MRA     | LFU   |              0.060 |
| 41.040 | 1.592 | MRA | MFU     | LFU   |              0.058 |
| 40.920 | 1.676 | MRU | MRA     | LRU   |              0.066 |
| 40.840 | 0.637 | MFU | MFU     | LRU   |              0.066 |
| 40.720 | 1.613 | MFU | MRU     | LFU   |              0.060 |
| 40.680 | 1.217 | MFU | MRA     | LRU   |              0.064 |
| 40.600 | 1.284 | MRA | MRA     | LRU   |              0.065 |
| 40.520 | 0.816 | MRU | MFU     | LRU   |              0.065 |
| 40.520 | 1.401 | MFU | MFU     | LFU   |              0.063 |
| 40.320 | 1.078 | MRU | MFU     | LFU   |              0.065 |
| 37.960 | 1.422 | MFU | MFU     | FIFO  |              0.060 |
| 37.760 | 1.835 | MRU | MRU     | FIFO  |              0.060 |
| 37.640 | 1.865 | MRU | MFU     | FIFO  |              0.063 |
| 37.480 | 0.601 | MRA | MFU     | FIFO  |              0.060 |
| 37.440 | 0.742 | MFU | MRA     | FIFO  |              0.061 |
| 37.400 | 2.028 | MRU | MRA     | FIFO  |              0.060 |
| 37.320 | 1.618 | MRA | MRU     | FIFO  |              0.063 |
| 36.920 | 2.218 | MFU | MRU     | FIFO  |              0.063 |
| 36.560 | 1.292 | MRA | MRA     | FIFO  |              0.063 |

#### Memory Size 512

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 44.480 | 1.230 | MRU | MRU     | LFU   |              0.063 |
| 44.480 | 1.230 | MRU | MRU     | FIFO  |              0.074 |
| 44.480 | 1.230 | MRU | MRU     | LRU   |              0.076 |
| 44.120 | 0.854 | MRA | MRU     | LFU   |              0.070 |
| 44.120 | 0.449 | MFU | MRA     | LFU   |              0.075 |
| 44.120 | 0.449 | MFU | MRA     | LRU   |              0.077 |
| 44.120 | 0.854 | MRA | MRU     | LRU   |              0.071 |
| 44.120 | 0.449 | MFU | MRA     | FIFO  |              0.067 |
| 44.080 | 0.900 | MRA | MRU     | FIFO  |              0.064 |
| 43.960 | 0.924 | MRU | MRA     | LRU   |              0.077 |
| 43.960 | 0.924 | MRU | MRA     | FIFO  |              0.064 |
| 43.960 | 0.924 | MRU | MRA     | LFU   |              0.076 |
| 43.840 | 1.274 | MRA | MFU     | LRU   |              0.076 |
| 43.840 | 1.274 | MRA | MFU     | LFU   |              0.074 |
| 43.840 | 1.274 | MRA | MFU     | FIFO  |              0.071 |
| 43.680 | 0.299 | MRU | MFU     | LRU   |              0.078 |
| 43.680 | 0.299 | MRU | MFU     | FIFO  |              0.071 |
| 43.680 | 0.299 | MRU | MFU     | LFU   |              0.077 |
| 43.680 | 1.662 | MFU | MRU     | LRU   |              0.077 |
| 43.680 | 1.662 | MFU | MRU     | FIFO  |              0.076 |
| 43.680 | 1.662 | MFU | MRU     | LFU   |              0.077 |
| 43.600 | 0.849 | MRA | MRA     | LRU   |              0.073 |
| 43.600 | 0.849 | MRA | MRA     | LFU   |              0.078 |
| 43.600 | 0.849 | MRA | MRA     | FIFO  |              0.073 |
| 43.240 | 0.543 | MFU | MFU     | LFU   |              0.070 |
| 43.240 | 0.543 | MFU | MFU     | FIFO  |              0.069 |
| 43.240 | 0.543 | MFU | MFU     | LRU   |              0.075 |

## Test

#### Memory Size 0

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 2.000 | 0.669 | MFU | MFU     | LFU   |              0.027 |
| 2.000 | 0.669 | MFU | MFU     | LRU   |              0.028 |
| 2.000 | 0.669 | MFU | MRA     | LFU   |              0.029 |
| 2.000 | 0.669 | MRA | MFU     | LFU   |              0.025 |
| 2.000 | 0.669 | MFU | MRA     | LRU   |              0.026 |
| 2.000 | 0.669 | MFU | MRU     | LRU   |              0.025 |
| 2.000 | 0.669 | MFU | MRU     | LFU   |              0.030 |
| 2.000 | 0.669 | MRA | MRU     | LFU   |              0.027 |
| 2.000 | 0.669 | MRA | MRA     | LRU   |              0.027 |
| 2.000 | 0.669 | MRA | MRA     | LFU   |              0.028 |
| 2.000 | 0.669 | MRA | MFU     | LRU   |              0.027 |
| 2.000 | 0.669 | MRU | MRU     | LRU   |              0.029 |
| 2.000 | 0.669 | MRU | MRU     | LFU   |              0.025 |
| 2.000 | 0.669 | MRU | MRA     | LRU   |              0.026 |
| 2.000 | 0.669 | MRU | MFU     | LFU   |              0.026 |
| 2.000 | 0.669 | MRA | MRU     | LRU   |              0.028 |
| 2.000 | 0.669 | MRU | MRA     | LFU   |              0.033 |
| 2.000 | 0.669 | MRU | MFU     | LRU   |              0.030 |
| 1.840 | 0.557 | MRA | MFU     | FIFO  |              0.030 |
| 1.840 | 0.557 | MFU | MRA     | FIFO  |              0.030 |
| 1.840 | 0.557 | MFU | MRU     | FIFO  |              0.028 |
| 1.840 | 0.557 | MFU | MFU     | FIFO  |              0.027 |
| 1.840 | 0.557 | MRA | MRA     | FIFO  |              0.030 |
| 1.840 | 0.557 | MRU | MFU     | FIFO  |              0.029 |
| 1.840 | 0.557 | MRA | MRU     | FIFO  |              0.034 |
| 1.840 | 0.557 | MRU | MRA     | FIFO  |              0.033 |
| 1.840 | 0.557 | MRU | MRU     | FIFO  |              0.028 |

#### Memory Size 2

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 2.680 | 0.652 | MRU | MRA     | FIFO  |              0.032 |
| 2.680 | 0.652 | MFU | MRA     | FIFO  |              0.028 |
| 2.560 | 0.528 | MRA | MRA     | FIFO  |              0.029 |
| 2.440 | 0.637 | MFU | MRU     | FIFO  |              0.033 |
| 2.440 | 0.637 | MRU | MRU     | FIFO  |              0.030 |
| 2.440 | 0.637 | MRU | MFU     | FIFO  |              0.033 |
| 2.440 | 0.637 | MFU | MFU     | FIFO  |              0.032 |
| 2.400 | 0.632 | MRA | MRU     | FIFO  |              0.030 |
| 2.400 | 0.632 | MRA | MFU     | FIFO  |              0.030 |
| 2.320 | 0.515 | MRA | MRA     | LFU   |              0.030 |
| 2.280 | 0.560 | MFU | MRA     | LFU   |              0.029 |
| 2.280 | 0.560 | MRU | MRA     | LFU   |              0.025 |
| 2.160 | 0.794 | MFU | MRA     | LRU   |              0.027 |
| 2.160 | 0.794 | MRU | MRA     | LRU   |              0.029 |
| 2.160 | 0.731 | MRA | MRA     | LRU   |              0.028 |
| 1.840 | 0.662 | MRU | MRU     | LRU   |              0.028 |
| 1.840 | 0.662 | MFU | MFU     | LRU   |              0.025 |
| 1.840 | 0.662 | MRA | MFU     | LRU   |              0.026 |
| 1.840 | 0.662 | MFU | MRU     | LRU   |              0.027 |
| 1.840 | 0.662 | MRU | MFU     | LRU   |              0.027 |
| 1.840 | 0.662 | MRA | MRU     | LRU   |              0.026 |
| 1.520 | 0.515 | MFU | MFU     | LFU   |              0.031 |
| 1.520 | 0.515 | MFU | MRU     | LFU   |              0.026 |
| 1.520 | 0.515 | MRU | MRU     | LFU   |              0.030 |
| 1.520 | 0.515 | MRU | MFU     | LFU   |              0.024 |
| 1.440 | 0.388 | MRA | MFU     | LFU   |              0.028 |
| 1.440 | 0.388 | MRA | MRU     | LFU   |              0.027 |

#### Memory Size 4

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 3.280 | 0.431 | MFU | MRA     | LRU   |              0.028 |
| 3.280 | 0.431 | MRU | MRA     | LRU   |              0.029 |
| 3.160 | 0.585 | MRA | MRA     | LRU   |              0.030 |
| 2.840 | 0.571 | MRU | MRU     | LRU   |              0.029 |
| 2.840 | 0.571 | MRU | MFU     | LRU   |              0.025 |
| 2.840 | 0.571 | MFU | MRU     | LRU   |              0.026 |
| 2.840 | 0.571 | MFU | MFU     | LRU   |              0.028 |
| 2.760 | 0.924 | MRA | MFU     | LFU   |              0.030 |
| 2.760 | 0.924 | MRA | MRU     | LFU   |              0.027 |
| 2.640 | 0.557 | MRA | MRA     | FIFO  |              0.030 |
| 2.560 | 0.916 | MFU | MFU     | LFU   |              0.027 |
| 2.560 | 0.916 | MRU | MRU     | LFU   |              0.027 |
| 2.560 | 0.445 | MRA | MFU     | FIFO  |              0.030 |
| 2.560 | 0.916 | MRU | MFU     | LFU   |              0.025 |
| 2.560 | 0.445 | MRA | MRU     | FIFO  |              0.033 |
| 2.560 | 0.916 | MFU | MRU     | LFU   |              0.027 |
| 2.440 | 0.662 | MRU | MRA     | LFU   |              0.026 |
| 2.440 | 0.662 | MFU | MRA     | LFU   |              0.028 |
| 2.400 | 0.335 | MFU | MFU     | FIFO  |              0.027 |
| 2.400 | 0.335 | MRU | MRU     | FIFO  |              0.031 |
| 2.400 | 0.335 | MFU | MRU     | FIFO  |              0.034 |
| 2.400 | 0.716 | MRA | MFU     | LRU   |              0.028 |
| 2.400 | 0.473 | MRA | MRA     | LFU   |              0.031 |
| 2.400 | 0.716 | MRA | MRU     | LRU   |              0.026 |
| 2.400 | 0.335 | MRU | MFU     | FIFO  |              0.030 |
| 1.960 | 0.637 | MFU | MRA     | FIFO  |              0.034 |
| 1.960 | 0.637 | MRU | MRA     | FIFO  |              0.034 |

#### Memory Size 8

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 3.920 | 0.560 | MRU | MRA     | LFU   |              0.028 |
| 3.920 | 0.560 | MFU | MRA     | LFU   |              0.031 |
| 3.800 | 0.522 | MRA | MRA     | LRU   |              0.029 |
| 3.760 | 0.344 | MRA | MRA     | LFU   |              0.026 |
| 3.640 | 0.843 | MRU | MRA     | LRU   |              0.029 |
| 3.640 | 0.843 | MFU | MRA     | LRU   |              0.026 |
| 3.440 | 0.804 | MRA | MRA     | FIFO  |              0.033 |
| 3.320 | 0.826 | MFU | MRA     | FIFO  |              0.033 |
| 3.280 | 0.744 | MRU | MRU     | LRU   |              0.026 |
| 3.280 | 0.744 | MRU | MFU     | LRU   |              0.029 |
| 3.240 | 0.852 | MRU | MRA     | FIFO  |              0.028 |
| 3.160 | 0.543 | MFU | MRU     | LRU   |              0.031 |
| 3.160 | 0.543 | MFU | MFU     | LRU   |              0.027 |
| 3.120 | 0.873 | MRA | MRU     | LRU   |              0.031 |
| 3.120 | 0.873 | MRA | MFU     | LRU   |              0.029 |
| 2.880 | 0.873 | MRA | MRU     | FIFO  |              0.035 |
| 2.880 | 0.873 | MRA | MFU     | FIFO  |              0.031 |
| 2.840 | 1.076 | MRA | MFU     | LFU   |              0.028 |
| 2.800 | 1.431 | MFU | MRU     | LFU   |              0.029 |
| 2.760 | 1.280 | MFU | MFU     | LFU   |              0.030 |
| 2.680 | 0.968 | MRA | MRU     | LFU   |              0.027 |
| 2.680 | 1.532 | MRU | MRU     | LFU   |              0.030 |
| 2.600 | 0.438 | MRU | MFU     | FIFO  |              0.031 |
| 2.600 | 1.409 | MRU | MFU     | LFU   |              0.030 |
| 2.600 | 0.438 | MRU | MRU     | FIFO  |              0.029 |
| 2.560 | 0.427 | MFU | MFU     | FIFO  |              0.035 |
| 2.560 | 0.427 | MFU | MRU     | FIFO  |              0.029 |

#### Memory Size 16

|  mean |   std | QA  | Explore | Evict | avg_duration_hours |
| ----: | ----: | :-- | :------ | :---- | -----------------: |
| 8.560 | 1.627 | MRA | MRA     | LFU   |              0.033 |
| 7.760 | 1.439 | MFU | MRA     | LFU   |              0.030 |
| 7.560 | 1.084 | MRU | MRA     | LFU   |              0.029 |
| 7.480 | 2.026 | MFU | MFU     | LFU   |              0.028 |
| 7.480 | 1.970 | MRA | MRU     | LFU   |              0.034 |
| 7.400 | 1.956 | MRU | MFU     | LFU   |              0.031 |
| 7.360 | 1.203 | MFU | MRU     | LFU   |              0.032 |
| 7.280 | 1.849 | MRA | MFU     | LFU   |              0.032 |
| 7.160 | 1.371 | MRU | MRU     | LFU   |              0.032 |
| 5.920 | 0.652 | MRA | MRA     | FIFO  |              0.035 |
| 5.760 | 0.650 | MRA | MRU     | FIFO  |              0.032 |
| 5.600 | 1.051 | MRA | MFU     | FIFO  |              0.034 |
| 5.440 | 1.203 | MRA | MRA     | LRU   |              0.033 |
| 5.360 | 1.007 | MRA | MRU     | LRU   |              0.030 |
| 5.360 | 1.183 | MRU | MRU     | LRU   |              0.030 |
| 5.320 | 0.531 | MFU | MFU     | FIFO  |              0.032 |
| 5.240 | 1.061 | MFU | MRA     | FIFO  |              0.037 |
| 5.200 | 0.522 | MRU | MFU     | FIFO  |              0.038 |
| 5.200 | 1.486 | MFU | MFU     | LRU   |              0.028 |
| 5.160 | 0.958 | MRU | MRA     | FIFO  |              0.037 |
| 5.080 | 0.796 | MRU | MRA     | LRU   |              0.032 |
| 5.040 | 0.408 | MFU | MRU     | FIFO  |              0.035 |
| 5.040 | 0.804 | MFU | MRA     | LRU   |              0.032 |
| 5.000 | 0.490 | MRU | MRU     | FIFO  |              0.035 |
| 4.920 | 0.786 | MRA | MFU     | LRU   |              0.029 |
| 4.800 | 1.368 | MFU | MRU     | LRU   |              0.034 |
| 4.720 | 1.163 | MRU | MFU     | LRU   |              0.034 |

#### Memory Size 32

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 15.520 | 1.642 | MRA | MRU     | LFU   |              0.031 |
| 15.040 | 1.141 | MRA | MFU     | LFU   |              0.038 |
| 14.800 | 1.507 | MFU | MFU     | LFU   |              0.034 |
| 14.720 | 0.601 | MRU | MFU     | LFU   |              0.032 |
| 14.480 | 3.367 | MFU | MRA     | LFU   |              0.034 |
| 14.240 | 2.114 | MRU | MRA     | LFU   |              0.034 |
| 14.040 | 2.681 | MRA | MRA     | LFU   |              0.036 |
| 13.720 | 2.204 | MRU | MRU     | LFU   |              0.032 |
| 13.600 | 2.055 | MFU | MRU     | LFU   |              0.038 |
| 10.320 | 1.300 | MFU | MRU     | LRU   |              0.036 |
| 10.280 | 1.354 | MRU | MRU     | LRU   |              0.035 |
| 10.120 | 1.093 | MFU | MRA     | LRU   |              0.034 |
|  9.800 | 0.780 | MRU | MRA     | LRU   |              0.039 |
|  9.680 | 1.287 | MFU | MFU     | LRU   |              0.032 |
|  9.520 | 0.863 | MRA | MFU     | FIFO  |              0.034 |
|  9.480 | 1.136 | MRU | MFU     | LRU   |              0.032 |
|  9.480 | 0.993 | MRA | MRU     | LRU   |              0.034 |
|  9.200 | 1.081 | MRA | MRA     | LRU   |              0.037 |
|  8.840 | 0.637 | MRU | MRA     | FIFO  |              0.037 |
|  8.800 | 0.972 | MRA | MFU     | LRU   |              0.031 |
|  8.800 | 1.020 | MFU | MFU     | FIFO  |              0.039 |
|  8.680 | 0.826 | MRU | MRU     | FIFO  |              0.034 |
|  8.480 | 0.816 | MFU | MRA     | FIFO  |              0.036 |
|  8.360 | 0.557 | MRU | MFU     | FIFO  |              0.036 |
|  8.280 | 0.993 | MRA | MRA     | FIFO  |              0.036 |
|  7.920 | 1.143 | MRA | MRU     | FIFO  |              0.038 |
|  7.800 | 0.820 | MFU | MRU     | FIFO  |              0.038 |

#### Memory Size 64

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 22.600 | 1.507 | MFU | MRU     | LFU   |              0.040 |
| 21.760 | 3.444 | MRA | MRA     | LFU   |              0.042 |
| 21.680 | 2.303 | MRU | MRU     | LFU   |              0.039 |
| 21.520 | 2.770 | MRA | MRU     | LFU   |              0.045 |
| 21.360 | 3.102 | MRA | MFU     | LFU   |              0.046 |
| 21.240 | 0.991 | MFU | MFU     | LFU   |              0.047 |
| 20.920 | 1.970 | MRU | MRA     | LFU   |              0.049 |
| 20.520 | 1.342 | MRU | MFU     | LFU   |              0.044 |
| 20.400 | 3.774 | MFU | MRA     | LFU   |              0.040 |
| 16.480 | 1.093 | MFU | MRU     | LRU   |              0.040 |
| 16.360 | 1.472 | MFU | MRA     | LRU   |              0.040 |
| 16.040 | 1.998 | MRU | MRU     | LRU   |              0.042 |
| 16.000 | 1.356 | MFU | MFU     | LRU   |              0.043 |
| 15.840 | 1.727 | MRU | MRA     | LRU   |              0.046 |
| 15.840 | 1.704 | MFU | MRU     | FIFO  |              0.041 |
| 15.720 | 2.802 | MRA | MRU     | LRU   |              0.047 |
| 15.640 | 0.612 | MRU | MFU     | LRU   |              0.041 |
| 15.320 | 0.917 | MRA | MRU     | FIFO  |              0.045 |
| 15.320 | 1.742 | MRA | MFU     | LRU   |              0.044 |
| 15.120 | 0.854 | MRU | MRA     | FIFO  |              0.047 |
| 15.080 | 1.107 | MRU | MRU     | FIFO  |              0.044 |
| 15.040 | 0.662 | MRA | MRA     | FIFO  |              0.044 |
| 14.680 | 1.700 | MRU | MFU     | FIFO  |              0.045 |
| 14.520 | 1.378 | MFU | MRA     | FIFO  |              0.044 |
| 14.000 | 1.683 | MFU | MFU     | FIFO  |              0.049 |
| 14.000 | 0.912 | MRA | MRA     | LRU   |              0.042 |
| 13.600 | 1.585 | MRA | MFU     | FIFO  |              0.043 |

#### Memory Size 128

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 33.200 | 1.595 | MRU | MRA     | LFU   |              0.054 |
| 33.040 | 1.957 | MFU | MRA     | LFU   |              0.055 |
| 32.920 | 1.324 | MFU | MFU     | LFU   |              0.057 |
| 32.840 | 2.236 | MRA | MRA     | LFU   |              0.056 |
| 32.360 | 1.704 | MRA | MFU     | LFU   |              0.053 |
| 32.040 | 1.248 | MRA | MRU     | LFU   |              0.064 |
| 31.920 | 1.994 | MRU | MFU     | LFU   |              0.051 |
| 31.880 | 3.381 | MRU | MRU     | LFU   |              0.058 |
| 31.520 | 2.637 | MRA | MRA     | LRU   |              0.056 |
| 31.360 | 1.472 | MRU | MFU     | LRU   |              0.059 |
| 31.040 | 1.755 | MFU | MRU     | LFU   |              0.057 |
| 30.840 | 1.556 | MFU | MRA     | LRU   |              0.055 |
| 30.840 | 1.546 | MFU | MFU     | LRU   |              0.057 |
| 30.560 | 1.493 | MFU | MRU     | LRU   |              0.059 |
| 30.480 | 2.613 | MRU | MRA     | LRU   |              0.057 |
| 30.440 | 0.958 | MRA | MRU     | LRU   |              0.051 |
| 30.320 | 1.737 | MRA | MFU     | LRU   |              0.059 |
| 29.680 | 2.317 | MRU | MRU     | LRU   |              0.052 |
| 25.760 | 1.038 | MRA | MFU     | FIFO  |              0.055 |
| 25.520 | 1.929 | MFU | MRA     | FIFO  |              0.052 |
| 25.400 | 0.955 | MFU | MFU     | FIFO  |              0.057 |
| 25.160 | 1.651 | MFU | MRU     | FIFO  |              0.062 |
| 25.160 | 2.072 | MRU | MRA     | FIFO  |              0.053 |
| 24.600 | 1.043 | MRU | MRU     | FIFO  |              0.053 |
| 24.480 | 0.917 | MRU | MFU     | FIFO  |              0.053 |
| 23.960 | 2.451 | MRA | MRA     | FIFO  |              0.056 |
| 23.800 | 1.635 | MRA | MRU     | FIFO  |              0.053 |

#### Memory Size 256

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 44.400 | 2.044 | MRU | MRA     | LFU   |              0.073 |
| 44.000 | 0.996 | MRA | MFU     | LFU   |              0.072 |
| 43.920 | 1.792 | MRU | MFU     | LFU   |              0.071 |
| 43.920 | 1.478 | MRU | MRA     | LRU   |              0.065 |
| 43.800 | 2.040 | MRA | MRA     | LRU   |              0.075 |
| 43.800 | 1.265 | MFU | MRA     | LFU   |              0.074 |
| 43.760 | 1.061 | MRA | MFU     | LRU   |              0.075 |
| 43.760 | 0.265 | MFU | MFU     | LFU   |              0.070 |
| 43.600 | 1.502 | MRU | MFU     | LRU   |              0.075 |
| 43.560 | 2.767 | MRA | MRA     | LFU   |              0.080 |
| 43.480 | 0.483 | MFU | MFU     | LRU   |              0.076 |
| 43.360 | 1.347 | MFU | MRA     | LRU   |              0.071 |
| 43.080 | 1.401 | MRU | MRU     | LRU   |              0.073 |
| 42.760 | 1.183 | MRA | MRU     | LFU   |              0.069 |
| 42.600 | 1.906 | MRU | MRU     | LFU   |              0.068 |
| 42.360 | 1.732 | MFU | MRU     | LRU   |              0.067 |
| 42.000 | 1.259 | MRA | MRU     | LRU   |              0.067 |
| 41.960 | 2.037 | MFU | MRU     | LFU   |              0.070 |
| 38.800 | 1.766 | MRU | MFU     | FIFO  |              0.063 |
| 38.640 | 1.076 | MRA | MFU     | FIFO  |              0.066 |
| 38.560 | 1.106 | MFU | MRA     | FIFO  |              0.066 |
| 38.400 | 1.425 | MRA | MRU     | FIFO  |              0.064 |
| 38.400 | 2.514 | MRU | MRA     | FIFO  |              0.060 |
| 38.240 | 2.584 | MRA | MRA     | FIFO  |              0.070 |
| 38.080 | 1.468 | MFU | MFU     | FIFO  |              0.065 |
| 37.960 | 1.546 | MRU | MRU     | FIFO  |              0.075 |
| 37.760 | 2.483 | MFU | MRU     | FIFO  |              0.063 |

#### Memory Size 512

|   mean |   std | QA  | Explore | Evict | avg_duration_hours |
| -----: | ----: | :-- | :------ | :---- | -----------------: |
| 46.520 | 1.896 | MRU | MRA     | LRU   |              0.089 |
| 46.520 | 1.896 | MRU | MRA     | LFU   |              0.087 |
| 46.440 | 1.891 | MRU | MRA     | FIFO  |              0.075 |
| 46.160 | 1.965 | MRA | MRA     | FIFO  |              0.086 |
| 46.160 | 1.965 | MRA | MRA     | LRU   |              0.077 |
| 46.160 | 1.965 | MRA | MRA     | LFU   |              0.082 |
| 46.120 | 1.457 | MFU | MRA     | LFU   |              0.082 |
| 46.120 | 1.457 | MFU | MRA     | LRU   |              0.084 |
| 46.040 | 1.439 | MFU | MRA     | FIFO  |              0.094 |
| 45.920 | 0.392 | MRA | MFU     | LFU   |              0.085 |
| 45.920 | 0.392 | MRA | MFU     | LRU   |              0.081 |
| 45.880 | 0.325 | MRA | MFU     | FIFO  |              0.078 |
| 45.720 | 1.552 | MFU | MFU     | LFU   |              0.085 |
| 45.720 | 1.552 | MFU | MFU     | LRU   |              0.080 |
| 45.680 | 1.262 | MRU | MFU     | LFU   |              0.087 |
| 45.680 | 1.262 | MRU | MFU     | LRU   |              0.077 |
| 45.680 | 1.588 | MFU | MFU     | FIFO  |              0.085 |
| 45.640 | 1.209 | MRU | MRU     | LRU   |              0.082 |
| 45.640 | 1.209 | MRU | MRU     | LFU   |              0.088 |
| 45.600 | 1.193 | MRU | MRU     | FIFO  |              0.085 |
| 45.520 | 1.287 | MRU | MFU     | FIFO  |              0.077 |
| 45.440 | 1.141 | MRA | MRU     | LRU   |              0.081 |
| 45.440 | 1.141 | MRA | MRU     | LFU   |              0.082 |
| 45.320 | 1.063 | MRA | MRU     | FIFO  |              0.081 |
| 45.280 | 1.489 | MFU | MRU     | LFU   |              0.087 |
| 45.280 | 1.489 | MFU | MRU     | LRU   |              0.081 |
| 45.280 | 1.489 | MFU | MRU     | FIFO  |              0.083 |

## Neural Agent

### Train

#### Memory Size 0

| architecture_type |  mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | ----: | ----: | :----- | :------ | :---- | -----------------: |
| lstm              | 8.000 | 0.000 | Neural | Neural  | FIFO  |              0.342 |
| transformer       | 8.000 | 0.000 | Neural | Neural  | FIFO  |              0.481 |

#### Memory Size 2

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 14.400 | 1.855 | Neural | Neural  | FIFO  |              0.585 |
| lstm              | 11.600 | 2.417 | Neural | Neural  | FIFO  |              0.436 |

#### Memory Size 4

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 15.200 | 0.748 | Neural | Neural  | FIFO  |              0.656 |
| lstm              | 10.800 | 1.833 | Neural | Neural  | FIFO  |              0.482 |

#### Memory Size 8

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 15.400 | 1.020 | Neural | Neural  | FIFO  |              0.781 |
| lstm              | 13.400 | 1.855 | Neural | Neural  | FIFO  |              0.598 |

#### Memory Size 16

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 15.600 | 1.356 | Neural | Neural  | FIFO  |              0.990 |
| lstm              | 11.600 | 1.020 | Neural | Neural  | FIFO  |              0.906 |

#### Memory Size 32

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 16.800 | 0.748 | Neural | Neural  | FIFO  |              1.432 |
| lstm              | 12.600 | 2.577 | Neural | Neural  | FIFO  |              1.344 |

#### Memory Size 64

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 16.000 | 1.265 | Neural | Neural  | FIFO  |              2.345 |
| lstm              |  9.800 | 0.980 | Neural | Neural  | FIFO  |              2.116 |

#### Memory Size 128

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 16.800 | 0.748 | Neural | Neural  | FIFO  |              3.971 |
| lstm              | 10.200 | 1.166 | Neural | Neural  | FIFO  |              3.785 |

#### Memory Size 256

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 16.000 | 2.449 | Neural | Neural  | FIFO  |              6.709 |
| lstm              | 11.000 | 0.632 | Neural | Neural  | FIFO  |              6.259 |

#### Memory Size 512

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 17.600 | 3.072 | Neural | Neural  | FIFO  |              9.115 |
| lstm              | 15.200 | 0.980 | Neural | Neural  | FIFO  |              8.931 |

### Test

#### Memory Size 0

| architecture_type |  mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | ----: | ----: | :----- | :------ | :---- | -----------------: |
| lstm              | 8.000 | 0.000 | Neural | Neural  | FIFO  |              0.001 |
| transformer       | 8.000 | 0.000 | Neural | Neural  | FIFO  |              0.000 |

#### Memory Size 2

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| lstm              | 11.000 | 2.683 | Neural | Neural  | FIFO  |              0.000 |
| transformer       | 10.000 | 3.033 | Neural | Neural  | FIFO  |              0.001 |

#### Memory Size 4

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 12.000 | 1.414 | Neural | Neural  | FIFO  |              0.000 |
| lstm              |  8.000 | 0.632 | Neural | Neural  | FIFO  |              0.001 |

#### Memory Size 8

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 13.800 | 1.600 | Neural | Neural  | FIFO  |              0.001 |
| lstm              | 13.400 | 1.356 | Neural | Neural  | FIFO  |              0.000 |

#### Memory Size 16

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 12.400 | 3.200 | Neural | Neural  | FIFO  |              0.000 |
| lstm              | 10.400 | 1.855 | Neural | Neural  | FIFO  |              0.000 |

#### Memory Size 32

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 14.000 | 1.673 | Neural | Neural  | FIFO  |              0.000 |
| lstm              |  8.800 | 5.307 | Neural | Neural  | FIFO  |              0.002 |

#### Memory Size 64

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 13.800 | 1.166 | Neural | Neural  | FIFO  |              0.001 |
| lstm              |  7.600 | 1.625 | Neural | Neural  | FIFO  |              0.002 |

#### Memory Size 128

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 11.800 | 3.187 | Neural | Neural  | FIFO  |              0.001 |
| lstm              |  7.600 | 2.332 | Neural | Neural  | FIFO  |              0.003 |

#### Memory Size 256

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| transformer       | 13.800 | 2.482 | Neural | Neural  | FIFO  |              0.002 |
| lstm              |  8.800 | 1.720 | Neural | Neural  | FIFO  |              0.002 |

#### Memory Size 512

| architecture_type |   mean |   std | QA     | Explore | Evict | avg_duration_hours |
| :---------------- | -----: | ----: | :----- | :------ | :---- | -----------------: |
| lstm              | 11.200 | 2.561 | Neural | Neural  | FIFO  |              0.003 |
| transformer       |  9.000 | 2.828 | Neural | Neural  | FIFO  |              0.002 |
