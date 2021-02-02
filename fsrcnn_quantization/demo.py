import os
import argparse
import cv2
import fsrcnn

SCALE_FACTOR = 4
SAVED_MODEL_PATH = 'fsrcnn_saved_model'
TFLITE_MODEL_PATH = 'fsrcnn_x4_ptq.tflite'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSRCNN Super Resolution Quantization Demo')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--quantize', dest='quantize', action='store_true')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--train-data', type=str, default='datasets/91-image_x4.h5')
    parser.add_argument('--saved-model-path', type=str, default='fsrcnn_saved_model')
    parser.add_argument('--quant-model-path', type=str, default='fsrcnn_x4_ptq.tflite')
    parser.add_argument('--eval-index', type=int, default=0)
    args = parser.parse_args()

    if args.train:
        print('TRAIN')
        if not os.path.exists(args.train_data):
            print(f'Train data not found: {args.train_data}')
            raise ValueError
        print(f'Preparing FSRCNN super resolution model with scale factor {SCALE_FACTOR}')
        model = fsrcnn.build(scale_factor=SCALE_FACTOR)
        print(f'Train with dataset {args.train_data}')
        model = fsrcnn.train(model, args.train_data, num_epochs=1, batch_size=64)
        print(f'Saving trained SavedModel to {SAVED_MODEL_PATH}')
        model.save(SAVED_MODEL_PATH, overwrite=True, include_optimizer=False, save_format='tf')
        
    elif args.quantize:
        print('QUANTIZE')
        print('Quantize floating point model to 8-bits using post training quantization')
        print(f'Loading SavedModel from {args.saved_model_path}')
        if not os.path.exists(args.saved_model_path):
            print(f'SavedModel not found at {args.saved_model_path}')
            raise ValueError
        print('Start quantization')
        quant_model = fsrcnn.quantize(args.saved_model_path)
        print(f'Saving quantized model to {TFLITE_MODEL_PATH}')
        open(TFLITE_MODEL_PATH, 'wb').write(quant_model)

    elif args.eval:
        print('EVALUATE')
        if not os.path.exists(args.quant_model_path):
            print(f'Quantized model file not found at {args.quant_model_path}')
            raise ValueError
        print(f'Evaluating image {args.eval_index} on {args.quant_model_path}')
        sr_y, psnr = fsrcnn.evaluate(args.quant_model_path, image_index=args.eval_index)
        print(f'Eval index {args.eval_index} psnr: {psnr:.2f}')
        output_file = f'sr_y_{args.eval_index}.png'
        print(f'Saving upscaled Y-channel to {output_file}') 
        cv2.imwrite(output_file, cv2.merge([sr_y, sr_y, sr_y]))
