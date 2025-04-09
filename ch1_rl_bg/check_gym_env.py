from gym import envs

if __name__ == "__main__":
    env_specs = envs.registry.all()
    envs_ids = [env_spec for env_spec in env_specs]
    print(envs_ids)