from gym import envs
import os.path as osp

print(osp.dirname(osp.realpath(__file__)))

envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
