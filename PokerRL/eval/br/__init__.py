from .LocalBRMaster import LocalBRMaster as LocalBRMaster

try:
    import ray
    from .DistBRMaster import DistBRMaster as DistBRMaster

except ImportError:
    pass
