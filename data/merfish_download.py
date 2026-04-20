from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from pathlib import Path

# path
download_base = Path('../data/abc_atlas')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

print("Downloading...")
abc_cache.get_file_path(
    directory='Zhuang-ABCA-1', 
    file_name='Zhuang-ABCA-1-log2'
)
print("Downloading complete!")