# Dataset locations and acquisition

## ReplicaCAD

Set these roots to local paths on the machine running the experiments:

```text
$WORLD_CACHE_DATA/replica_cad
$WORLD_CACHE_DATA/replica_cad_baked_lighting
```

The interactive dataset was validated with:

```text
scene dataset config: replicaCAD.scene_dataset_config.json
scene: configs/scenes/apt_1.scene_instance.json
rigid objects loaded: 120
```

Use:

```bash
export WORLD_CACHE_DATA=/path/to/worldcache_datasets
export REPLICACAD_ROOT="$WORLD_CACHE_DATA/replica_cad"
export REPLICACAD_BAKED_ROOT="$WORLD_CACHE_DATA/replica_cad_baked_lighting"
```

## 3RScan

The official downloader supplied by the dataset author reports approximately 94GB for the full release. Begin with one reference/rescan pair:

```text
reference: 02b33dfb-be2b-2d54-92d2-cd012b2b3c40
rescan:    fcf66d9e-622d-291c-84c2-bb23dfe31327
split:     train
```

Local root:

```text
$WORLD_CACHE_DATA/3rscan
```

The full 3RScan release is subject to the dataset provider's terms. Each user
must obtain the data and accept the applicable terms independently. Preserve
the official reference/rescan relationship and global transformation from
`3RScan.json`.
