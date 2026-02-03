import sys
import argparse
from rknn.api import RKNN
import os


def parse_arg():
    parser = argparse.ArgumentParser(description='Convert ONNX model to RKNN format')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the ONNX model file')
    parser.add_argument('--platform', type=str, required=True,
                        choices=['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588', 'rv1126b', 'rv1109', 'rv1126', 'rk1808'],
                        help='Target platform. Choose from: [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]')
    parser.add_argument('--dtype', type=str, default='i8',
                        choices=['i8', 'u8', 'fp'],
                        help='Data type for quantization (i8/u8 for quantized, fp for float). '
                             'Choose [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]; '
                             'Choose [u8, fp] for [rv1109, rv1126, rk1808]. Default: i8')
    parser.add_argument('--rknn-path', type=str, default=None,
                        help='Output path for RKNN model. Default: ./<model_name>.rknn')
    parser.add_argument('--data-path', type=str, default="datasets/COCO/coco_subset_20.txt",
                        help=f'Path to dataset file for quantization. Default: datasets/COCO/coco_subset_20.txt')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for RKNN model. Default: 1')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    platform = args.platform
    dataset_path = args.data_path
    batch_size = args.batch_size
    # Determine quantization based on dtype
    do_quant = args.dtype in ['i8', 'u8']
    
    # Determine output path
    if args.rknn_path:
        output_path = args.rknn_path
    else:
        output_path = os.path.join('./', os.path.basename(model_path).split('.')[0] + '.rknn')
    
    return model_path, platform, do_quant, output_path, dataset_path, batch_size

if __name__ == '__main__':
    model_path, platform, do_quant, output_path, dataset_path, batch_size = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path, rknn_batch_size=batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    print('rknn model saved to: ', output_path)

    # Release
    rknn.release()
