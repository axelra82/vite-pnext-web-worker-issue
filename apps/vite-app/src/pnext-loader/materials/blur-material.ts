import { ShaderMaterial, Texture } from 'three';
import { IUniform } from './types';

// see http://john-chapman-graphics.blogspot.co.at/2013/01/ssao-tutorial.html

export interface IBlurMaterialUniforms {
  [name: string]: IUniform<any>;
  screenWidth: IUniform<number>;
  screenHeight: IUniform<number>;
  map: IUniform<Texture | null>;
}

export class BlurMaterial extends ShaderMaterial {
  // @ts-ignore
  vertexShader = require('./shaders/blur.vert');
  // @ts-ignore
  fragmentShader = require('./shaders/blur.frag');
  uniforms: IBlurMaterialUniforms = {
    screenWidth: { type: 'f', value: 0 },
    screenHeight: { type: 'f', value: 0 },
    map: { type: 't', value: null },
  };
}
