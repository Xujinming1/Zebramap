import torch

MIN_DEPTH_EVAL = 1e-3

def compute_metrics(pred, depth, args):

    max_depth_eval = args.max_depth

    if 'hypersim' in args.test_data_path:
        max_depth_eval = 80
    elif 'sunrgbd' in args.test_data_path:
        max_depth_eval = 8

    pred[pred > max_depth_eval] = max_depth_eval
    depth[torch.isinf(depth)] = max_depth_eval

    pred[pred < MIN_DEPTH_EVAL] = MIN_DEPTH_EVAL
    depth[torch.isnan(depth)] = MIN_DEPTH_EVAL

    valid_mask = torch.logical_and(depth > MIN_DEPTH_EVAL, depth < max_depth_eval)
    
    pred, depth = pred[valid_mask], depth[valid_mask]

    if depth.numel() == 0:
        print('ok')
        return None
    
    thresh = torch.max((depth / pred), (pred / depth))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - depth
    diff_log = torch.log(pred) - torch.log(depth)

    abs_rel = torch.mean(torch.abs(diff) / depth)
    sq_rel = torch.mean(torch.pow(diff, 2) / depth)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(depth)))
    
    silog = torch.sqrt(torch.mean(diff_log ** 2) - torch.mean(diff_log) ** 2) * 100

    return {'d1': d1, 'd2': d2, 'd3': d3, 'abs_rel': abs_rel,
            'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 
            'log10':log10, 'silog':silog}
