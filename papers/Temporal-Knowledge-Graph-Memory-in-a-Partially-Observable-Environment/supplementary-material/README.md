# Supplementary material

## Environment source code

- `env.pdf`

## Environment analysis

- `env-analysis.pdf`

## The schematic of the environment hidden states from `t=0` to `t=99`

- `bird-eye-view.pdf`
- Random exploration actions were taken for demonstration.

## The knowledge graph view of the environment hidden states from `t=0` to `t=99`

- `knowledge-graph-view.pdf`
- Random exploration actions were taken for demonstration.

## The 9 plots of the 27 combinations of the sub-policies on the test env

- `symbolic-sub-policies.pdf`

## The evolution of the neuro-symbolic agent's memory from `t=0` to `t=99`

- `memory-evolution.pdf`

## The Q-values of the neuro-symbolic agent's from `t=0` to `t=99` on the test env

- `q_values.pdf`

## Hardware and software specs used in the experiments

- CPU: AMD Ryzen 9 7950X 16-Core Processor
- Memory: 128 GB
- Ubuntu 22.04 (Linux kernel 6.12.10)

Everything was run on the CPU

## Hyperparameters used for the RDF-Star-NS agent

- batch_size = 32
- terminates_at = 99
- num_episodes = 200
- num_iterations = 20,000
- target_update_interval = 50
- linear epsilon_decay_until = 10,000
- warm_start = 2,000
- optimizer = adam (learning rate=1e-4)
- gamma (discount factor) = 0.95

It took about 4 days, to train one RDF-Star-NS agent with long-term memory
capacity=512. The training time is not 100% accurate, since multiple trainings were done
simultaneously, and also the RDFLib package used added a lot of overhead.

### Number of parameters of the DQN

#### Neural Agent

#### LSTM

11573

#### Transformer

13781

### TKG-NS Agent

#### GCN

```yaml
total: 12316
architecture: gcn
gcn_layers: 544
entity_embeddings: 6208
relation_embeddings: 480
attention_aggregator_forget: 560
mlp_forget: 612
attention_aggregator_qa: 560
question_mlp: 1056
mlp_qa: 1124
attention_aggregator_explore: 560
mlp_explore: 612
```

#### Stare-GCN

```yaml
total: 14460
gcn_layers: 2688
entity_embeddings: 6208
relation_embeddings: 480
attention_aggregator_forget: 560
mlp_forget: 612
attention_aggregator_qa: 560
question_mlp: 1056
mlp_qa: 1124
attention_aggregator_explore: 560
mlp_explore: 612
```

#### Transformer

```yaml
total: 20492
tokenizer_entity_embeddings: 6208
tokenizer_relation_embeddings: 240
tokenizer_qualifier_mlp: 800
tokenizer_qualifier_attention: 560
tokenizer_token_projection: 1040
transformer_encoder: 6560
attention_aggregator_forget: 560
mlp_forget: 612
attention_aggregator_qa: 560
question_mlp: 1056
mlp_qa: 1124
attention_aggregator_explore: 560
mlp_explore: 612
```
