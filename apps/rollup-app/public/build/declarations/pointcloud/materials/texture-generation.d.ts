import { Color, Texture } from 'three';
import { IClassification, IGradient } from './types';
export declare function generateDataTexture(width: number, height: number, color: Color): Texture;
export declare function generateGradientTexture(gradient: IGradient): Texture;
export declare function generateClassificationTexture(classification: IClassification): Texture;
