# Seed a graph world from a Neuro-SAN network

A Neuro-SAN agent network is a HOCON file that lists agents (`tools`) and the agents each one may call. TerraLingua can turn such a file into a graph world without running the Neuro-SAN runtime: each local `tools` entry becomes one being and one node, each local tool reference becomes an edge, and each entry's `instructions` become that being's personality sentence.

```bash
terralingua \
  --graph.agent_network_hocon_path demo/local/neuro_san_network.hocon \
  --graph.agent_network_bidirectional_edges \
  --genome sentence_directed \
  --no-food_mechanism --init_food 0 --reproduction_cost -1
```

`graph.agent_network_hocon_path` switches the run to the graph world with the `agent_network` topology and requires `genome: sentence_directed`, because the personality comes from the file. `graph.agent_network_bidirectional_edges` adds the reverse of every edge.

`reproduction_cost` controls births: `-1` disables them, `0` makes them free (the child starts with `init_agent_energy`), and a positive value is taken from the parent and becomes the child's energy. A failed attempt still costs that value. A parent may also give the child extra energy; it is taken only when the birth succeeds.

`demo/local/neuro_san_network.hocon` is a small example network. The local Docker demo can start from it with `./demo/local/run.sh --hocon`.
