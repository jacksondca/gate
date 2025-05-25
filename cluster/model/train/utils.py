def get_model_name(params):
    kvs = [params['name']]
    for k, v in params.items():
        if k != 'name':
            kvs.append(f"{k}{v}")

    return "_".join(kvs)
