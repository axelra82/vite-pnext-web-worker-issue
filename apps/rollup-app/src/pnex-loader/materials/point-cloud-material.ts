import {
  AdditiveBlending,
  BufferGeometry,
  Camera,
  Color,
  LessEqualDepth,
  Material,
  NearestFilter,
  NoBlending,
  PerspectiveCamera,
  RawShaderMaterial,
  Scene,
  Texture,
  Vector2,
  Vector3,
  Vector4,
  WebGLRenderer,
} from 'three';
import {
  DEFAULT_HIGHLIGHT_COLOR,
  DEFAULT_MAX_POINT_SIZE,
  DEFAULT_MIN_POINT_SIZE,
  DEFAULT_RGB_BRIGHTNESS,
  DEFAULT_RGB_CONTRAST,
  DEFAULT_RGB_GAMMA,
  PERSPECTIVE_CAMERA,
} from '../constants';
import { PointCloudOctree } from '../point-cloud-octree';
import { PointCloudOctreeNode } from '../point-cloud-octree-node';
import { byLevelAndIndex } from '../utils/utils';
import { DEFAULT_CLASSIFICATION } from './classification';
import { ClipMode, IClipBox } from './clipping';
import {
  NormalFilteringMode,
  PointCloudMixingMode,
  PointColorType,
  PointOpacityType,
  PointShape,
  PointSizeType,
  TreeType,
} from './enums';
import { SPECTRAL } from './gradients';
import {
  generateClassificationTexture,
  generateDataTexture,
  generateGradientTexture,
} from './texture-generation';
import { IClassification, IGradient, IUniform } from './types';

export interface IPointCloudMaterialParameters {
  size: number;
  minSize: number;
  maxSize: number;
  treeType: TreeType;
  colorRgba: boolean;
}

export interface IPointCloudMaterialUniforms {
  bbSize: IUniform<[number, number, number]>;
  blendDepthSupplement: IUniform<number>;
  blendHardness: IUniform<number>;
  classificationLUT: IUniform<Texture>;
  clipBoxCount: IUniform<number>;
  clipBoxes: IUniform<Float32Array>;
  clipExtent: IUniform<[number, number, number, number]>;
  depthMap: IUniform<Texture | null>;
  diffuse: IUniform<[number, number, number]>;
  fov: IUniform<number>;
  gradient: IUniform<Texture>;
  heightMax: IUniform<number>;
  heightMin: IUniform<number>;
  intensityBrightness: IUniform<number>;
  intensityContrast: IUniform<number>;
  intensityGamma: IUniform<number>;
  intensityRange: IUniform<[number, number]>;
  level: IUniform<number>;
  maxSize: IUniform<number>;
  minSize: IUniform<number>;
  octreeSize: IUniform<number>;
  opacity: IUniform<number>;
  pcIndex: IUniform<number>;
  rgbBrightness: IUniform<number>;
  rgbContrast: IUniform<number>;
  rgbGamma: IUniform<number>;
  screenHeight: IUniform<number>;
  screenWidth: IUniform<number>;
  size: IUniform<number>;
  spacing: IUniform<number>;
  toModel: IUniform<number[]>;
  transition: IUniform<number>;
  uColor: IUniform<Color>;
  visibleNodes: IUniform<Texture>;
  vnStart: IUniform<number>;
  wClassification: IUniform<number>;
  wElevation: IUniform<number>;
  wIntensity: IUniform<number>;
  wReturnNumber: IUniform<number>;
  wRGB: IUniform<number>;
  wSourceID: IUniform<number>;
  opacityAttenuation: IUniform<number>;
  filterByNormalThreshold: IUniform<number>;
  highlightedPointCoordinate: IUniform<Vector3>;
  highlightedPointColor: IUniform<Vector4>;
  enablePointHighlighting: IUniform<boolean>;
  highlightedPointScale: IUniform<number>;
  normalFilteringMode: IUniform<number>;
  backgroundMap: IUniform<Texture | null>;
  pointCloudID: IUniform<number>;
  pointCloudMixAngle: IUniform<number>;
  stripeDistanceX: IUniform<number>;
  stripeDistanceY: IUniform<number>;
  stripeDivisorX: IUniform<number>;
  stripeDivisorY: IUniform<number>;
  pointCloudMixingMode: IUniform<number>;
}

const TREE_TYPE_DEFS = {
  [TreeType.OCTREE]: 'tree_type_octree',
  [TreeType.KDTREE]: 'tree_type_kdtree',
};

const SIZE_TYPE_DEFS = {
  [PointSizeType.FIXED]: 'fixed_point_size',
  [PointSizeType.ATTENUATED]: 'attenuated_point_size',
  [PointSizeType.ADAPTIVE]: 'adaptive_point_size',
};

const OPACITY_DEFS = {
  [PointOpacityType.ATTENUATED]: 'attenuated_opacity',
  [PointOpacityType.FIXED]: 'fixed_opacity',
};

const SHAPE_DEFS = {
  [PointShape.SQUARE]: 'square_point_shape',
  [PointShape.CIRCLE]: 'circle_point_shape',
  [PointShape.PARABOLOID]: 'paraboloid_point_shape',
};

const COLOR_DEFS = {
  [PointColorType.RGB]: 'color_type_rgb',
  [PointColorType.COLOR]: 'color_type_color',
  [PointColorType.DEPTH]: 'color_type_depth',
  [PointColorType.HEIGHT]: 'color_type_height',
  [PointColorType.INTENSITY]: 'color_type_intensity',
  [PointColorType.INTENSITY_GRADIENT]: 'color_type_intensity_gradient',
  [PointColorType.LOD]: 'color_type_lod',
  [PointColorType.POINT_INDEX]: 'color_type_point_index',
  [PointColorType.CLASSIFICATION]: 'color_type_classification',
  [PointColorType.RETURN_NUMBER]: 'color_type_return_number',
  [PointColorType.SOURCE]: 'color_type_source',
  [PointColorType.NORMAL]: 'color_type_normal',
  [PointColorType.PHONG]: 'color_type_phong',
  [PointColorType.RGB_HEIGHT]: 'color_type_rgb_height',
  [PointColorType.COMPOSITE]: 'color_type_composite',
};

const CLIP_MODE_DEFS = {
  [ClipMode.DISABLED]: 'clip_disabled',
  [ClipMode.CLIP_OUTSIDE]: 'clip_outside',
  [ClipMode.HIGHLIGHT_INSIDE]: 'clip_highlight_inside',
  [ClipMode.CLIP_HORIZONTALLY]: 'clip_horizontally',
  [ClipMode.CLIP_VERTICALLY]: 'clip_vertically',
};

export class PointCloudMaterial extends RawShaderMaterial {
  private static helperVec3 = new Vector3();
  private static helperVec2 = new Vector2();

  /**
   * Use the drawing buffer size instead of the dom client width and height when passing the screen height and screen width uniforms to the
   * shader. This is useful if you have offscreen canvases (which in some browsers return 0 as client width and client height).
   */
  useDrawingBufferSize = false;
  lights = false;
  fog = false;
  colorRgba = false;
  numClipBoxes: number = 0;
  clipBoxes: IClipBox[] = [];
  visibleNodesTexture: Texture | undefined;
  private visibleNodeTextureOffsets = new Map<string, number>();

  private _gradient = SPECTRAL;
  private gradientTexture: Texture | undefined = generateGradientTexture(this._gradient);

  private _classification: IClassification = DEFAULT_CLASSIFICATION;
  private classificationTexture: Texture | undefined = generateClassificationTexture(
    this._classification,
  );

  uniforms: IPointCloudMaterialUniforms & Record<string, IUniform<any>> = {
    bbSize: makeUniform('fv', [0, 0, 0] as [number, number, number]),
    blendDepthSupplement: makeUniform('f', 0.0),
    blendHardness: makeUniform('f', 2.0),
    classificationLUT: makeUniform('t', this.classificationTexture || new Texture()),
    clipBoxCount: makeUniform('f', 0),
    clipBoxes: makeUniform('Matrix4fv', [] as any),
    clipExtent: makeUniform('fv', [0.0, 0.0, 1.0, 1.0] as [number, number, number, number]),
    depthMap: makeUniform('t', null),
    diffuse: makeUniform('fv', [1, 1, 1] as [number, number, number]),
    fov: makeUniform('f', 1.0),
    gradient: makeUniform('t', this.gradientTexture || new Texture()),
    heightMax: makeUniform('f', 1.0),
    heightMin: makeUniform('f', 0.0),
    intensityBrightness: makeUniform('f', 0),
    intensityContrast: makeUniform('f', 0),
    intensityGamma: makeUniform('f', 1),
    intensityRange: makeUniform('fv', [0, 65000] as [number, number]),
    isLeafNode: makeUniform('b', 0),
    level: makeUniform('f', 0.0),
    maxSize: makeUniform('f', DEFAULT_MAX_POINT_SIZE),
    minSize: makeUniform('f', DEFAULT_MIN_POINT_SIZE),
    octreeSize: makeUniform('f', 0),
    opacity: makeUniform('f', 1.0),
    pcIndex: makeUniform('f', 0),
    rgbBrightness: makeUniform('f', DEFAULT_RGB_BRIGHTNESS),
    rgbContrast: makeUniform('f', DEFAULT_RGB_CONTRAST),
    rgbGamma: makeUniform('f', DEFAULT_RGB_GAMMA),
    screenHeight: makeUniform('f', 1.0),
    screenWidth: makeUniform('f', 1.0),
    size: makeUniform('f', 1),
    spacing: makeUniform('f', 1.0),
    toModel: makeUniform('Matrix4f', []),
    transition: makeUniform('f', 0.5),
    uColor: makeUniform('c', new Color(0xffffff)),
    // @ts-ignore
    visibleNodes: makeUniform('t', this.visibleNodesTexture || new Texture()),
    vnStart: makeUniform('f', 0.0),
    wClassification: makeUniform('f', 0),
    wElevation: makeUniform('f', 0),
    wIntensity: makeUniform('f', 0),
    wReturnNumber: makeUniform('f', 0),
    wRGB: makeUniform('f', 1),
    wSourceID: makeUniform('f', 0),
    opacityAttenuation: makeUniform('f', 1),
    filterByNormalThreshold: makeUniform('f', 0),
    highlightedPointCoordinate: makeUniform('fv', new Vector3()),
    highlightedPointColor: makeUniform('fv', DEFAULT_HIGHLIGHT_COLOR.clone()),
    enablePointHighlighting: makeUniform('b', true),
    highlightedPointScale: makeUniform('f', 2.0),
    backgroundMap: makeUniform('t', null),
    normalFilteringMode: makeUniform('i', NormalFilteringMode.ABSOLUTE_NORMAL_FILTERING_MODE),
    pointCloudID: makeUniform('f', 2),
    pointCloudMixingMode: makeUniform('i', PointCloudMixingMode.CHECKBOARD),
    stripeDistanceX: makeUniform('f', 5),
    stripeDistanceY: makeUniform('f', 5),
    stripeDivisorX: makeUniform('f', 2),
    stripeDivisorY: makeUniform('f', 2),
    pointCloudMixAngle: makeUniform('f', 31),
  };

  @uniform('bbSize') bbSize!: [number, number, number];
  @uniform('clipExtent') clipExtent!: [number, number, number, number];
  @uniform('depthMap') depthMap!: Texture | undefined;
  @uniform('fov') fov!: number;
  @uniform('heightMax') heightMax!: number;
  @uniform('heightMin') heightMin!: number;
  @uniform('intensityBrightness') intensityBrightness!: number;
  @uniform('intensityContrast') intensityContrast!: number;
  @uniform('intensityGamma') intensityGamma!: number;
  @uniform('intensityRange') intensityRange!: [number, number];
  @uniform('maxSize') maxSize!: number;
  @uniform('minSize') minSize!: number;
  @uniform('octreeSize') octreeSize!: number;
  //@uniform('opacity', true) opacity!: number;
  @uniform('rgbBrightness', true) rgbBrightness!: number;
  @uniform('rgbContrast', true) rgbContrast!: number;
  @uniform('rgbGamma', true) rgbGamma!: number;
  @uniform('screenHeight') screenHeight!: number;
  @uniform('screenWidth') screenWidth!: number;
  @uniform('size') size!: number;
  @uniform('spacing') spacing!: number;
  @uniform('transition') transition!: number;
  @uniform('uColor') color!: Color;
  @uniform('wClassification') weightClassification!: number;
  @uniform('wElevation') weightElevation!: number;
  @uniform('wIntensity') weightIntensity!: number;
  @uniform('wReturnNumber') weightReturnNumber!: number;
  @uniform('wRGB') weightRGB!: number;
  @uniform('wSourceID') weightSourceID!: number;
  @uniform('opacityAttenuation') opacityAttenuation!: number;
  @uniform('filterByNormalThreshold') filterByNormalThreshold!: number;
  @uniform('highlightedPointCoordinate') highlightedPointCoordinate!: Vector3;
  @uniform('highlightedPointColor') highlightedPointColor!: Vector4;
  @uniform('enablePointHighlighting') enablePointHighlighting!: boolean;
  @uniform('highlightedPointScale') highlightedPointScale!: number;
  @uniform('normalFilteringMode') normalFilteringMode!: number;
  @uniform('backgroundMap') backgroundMap!: Texture | undefined;
  @uniform('pointCloudID') pointCloudID!: number;
  @uniform('pointCloudMixingMode') pointCloudMixingMode!: number;
  @uniform('stripeDistanceX') stripeDistanceX!: number;
  @uniform('stripeDistanceY') stripeDistanceY!: number;
  @uniform('stripeDivisorX') stripeDivisorX!: number;
  @uniform('stripeDivisorY') stripeDivisorY!: number;
  @uniform('pointCloudMixAngle') pointCloudMixAngle!: number;

  @requiresShaderUpdate() useClipBox: boolean = false;
  @requiresShaderUpdate() weighted: boolean = false;
  @requiresShaderUpdate() pointColorType: PointColorType = PointColorType.RGB;
  @requiresShaderUpdate() pointSizeType: PointSizeType = PointSizeType.ADAPTIVE;
  @requiresShaderUpdate() clipMode: ClipMode = ClipMode.DISABLED;
  @requiresShaderUpdate() useEDL: boolean = false;
  @requiresShaderUpdate() shape: PointShape = PointShape.SQUARE;
  @requiresShaderUpdate() treeType: TreeType = TreeType.OCTREE;
  @requiresShaderUpdate() pointOpacityType: PointOpacityType = PointOpacityType.FIXED;
  @requiresShaderUpdate() useFilterByNormal: boolean = false;
  @requiresShaderUpdate() useTextureBlending: boolean = false;
  @requiresShaderUpdate() usePointCloudMixing: boolean = false;
  @requiresShaderUpdate() highlightPoint: boolean = false;

  attributes = {
    position: { type: 'fv', value: [] },
    color: { type: 'fv', value: [] },
    normal: { type: 'fv', value: [] },
    intensity: { type: 'f', value: [] },
    classification: { type: 'f', value: [] },
    returnNumber: { type: 'f', value: [] },
    numberOfReturns: { type: 'f', value: [] },
    pointSourceID: { type: 'f', value: [] },
    indices: { type: 'fv', value: [] },
  };

  constructor(parameters: Partial<IPointCloudMaterialParameters> = {}) {
    super();

    const tex = (this.visibleNodesTexture = generateDataTexture(2048, 1, new Color(0xffffff)));
    tex.minFilter = NearestFilter;
    tex.magFilter = NearestFilter;
    this.setUniform('visibleNodes', tex);

    this.treeType = getValid(parameters.treeType, TreeType.OCTREE);
    this.size = getValid(parameters.size, 1.0);
    this.minSize = getValid(parameters.minSize, 2.0);
    this.maxSize = getValid(parameters.maxSize, 50.0);

    this.colorRgba = Boolean(parameters.colorRgba);

    this.classification = DEFAULT_CLASSIFICATION;

    this.defaultAttributeValues.normal = [0, 0, 0];
    this.defaultAttributeValues.classification = [0, 0, 0];
    this.defaultAttributeValues.indices = [0, 0, 0, 0];

    this.vertexColors = true;

    this.updateShaderSource();
  }

  dispose(): void {
    super.dispose();

    if (this.gradientTexture) {
      this.gradientTexture.dispose();
      this.gradientTexture = undefined;
    }

    if (this.visibleNodesTexture) {
      this.visibleNodesTexture.dispose();
      this.visibleNodesTexture = undefined;
    }

    this.clearVisibleNodeTextureOffsets();

    if (this.classificationTexture) {
      this.classificationTexture.dispose();
      this.classificationTexture = undefined;
    }

    if (this.depthMap) {
      this.depthMap.dispose();
      this.depthMap = undefined;
    }
    if (this.backgroundMap) {
      this.backgroundMap.dispose();
      this.backgroundMap = undefined;
    }
  }

  clearVisibleNodeTextureOffsets(): void {
    this.visibleNodeTextureOffsets.clear();
  }

  updateShaderSource(): void {
    const vertSource = `precision highp float;
    precision highp int;
    
    #define max_clip_boxes 30
    
    attribute vec3 position;
    attribute vec3 color;
    
    #ifdef color_rgba
      attribute vec4 rgba;
    #endif
    
    attribute vec3 normal;
    attribute float intensity;
    attribute float classification;
    attribute float returnNumber;
    attribute float numberOfReturns;
    attribute float pointSourceID;
    attribute vec4 indices;
    attribute vec2 uv;
    
    uniform mat4 modelMatrix;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    uniform mat4 viewMatrix;
    uniform mat3 normalMatrix;
    
    uniform float pcIndex;
    
    uniform float screenWidth;
    uniform float screenHeight;
    uniform float fov;
    uniform float spacing;
    
    #if defined use_clip_box
      uniform mat4 clipBoxes[max_clip_boxes];
    #endif
    
    uniform float heightMin;
    uniform float heightMax;
    uniform float size; // pixel size factor
    uniform float minSize; // minimum pixel size
    uniform float maxSize; // maximum pixel size
    uniform float octreeSize;
    uniform vec3 bbSize;
    uniform vec3 uColor;
    uniform float opacity;
    uniform float clipBoxCount;
    uniform float level;
    uniform float vnStart;
    uniform bool isLeafNode;
    
    uniform float filterByNormalThreshold;
    uniform vec2 intensityRange;
    uniform float opacityAttenuation;
    uniform float intensityGamma;
    uniform float intensityContrast;
    uniform float intensityBrightness;
    uniform float rgbGamma;
    uniform float rgbContrast;
    uniform float rgbBrightness;
    uniform float transition;
    uniform float wRGB;
    uniform float wIntensity;
    uniform float wElevation;
    uniform float wClassification;
    uniform float wReturnNumber;
    uniform float wSourceID;
    
    uniform sampler2D visibleNodes;
    uniform sampler2D gradient;
    uniform sampler2D classificationLUT;
    uniform sampler2D depthMap;
    
    #ifdef use_texture_blending
      uniform sampler2D backgroundMap;
    #endif
    
    #ifdef use_point_cloud_mixing
      uniform int pointCloudMixingMode;
      uniform float pointCloudID;
    
      uniform float pointCloudMixAngle;
      uniform float stripeDistanceX;
      uniform float stripeDistanceY;
    
      uniform float stripeDivisorX;
      uniform float stripeDivisorY;
    #endif
    
    #ifdef highlight_point
      uniform vec3 highlightedPointCoordinate;
      uniform bool enablePointHighlighting;
      uniform float highlightedPointScale;
    #endif
    
    #ifdef use_filter_by_normal
      uniform int normalFilteringMode;
    #endif
    
    varying vec3 vColor;
    
    #if !defined(color_type_point_index)
      varying float vOpacity;
    #endif
    
    #if defined(weighted_splats)
      varying float vLinearDepth;
    #endif
    
    #if !defined(paraboloid_point_shape) && defined(use_edl)
      varying float vLogDepth;
    #endif
    
    #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0) || defined(paraboloid_point_shape)
      varying vec3 vViewPosition;
    #endif
    
    #if defined(weighted_splats) || defined(paraboloid_point_shape)
      varying float vRadius;
    #endif
    
    #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0)
      varying vec3 vNormal;
    #endif
    
    #ifdef highlight_point
      varying float vHighlight;
    #endif
    
    // ---------------------
    // OCTREE
    // ---------------------
    
    #if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_octree)
    
    /**
     * Rounds the specified number to the closest integer.
     */
    float round(float number){
      return floor(number + 0.5);
    }
    
    /**
     * Gets the number of 1-bits up to inclusive index position.
     *
     * number is treated as if it were an integer in the range 0-255
     */
    int numberOfOnes(int number, int index) {
      int numOnes = 0;
      int tmp = 128;
      for (int i = 7; i >= 0; i--) {
    
        if (number >= tmp) {
          number = number - tmp;
    
          if (i <= index) {
            numOnes++;
          }
        }
    
        tmp = tmp / 2;
      }
    
      return numOnes;
    }
    
    /**
     * Checks whether the bit at index is 1.0
     *
     * number is treated as if it were an integer in the range 0-255
     */
    bool isBitSet(int number, int index){
    
      // weird multi else if due to lack of proper array, int and bitwise support in WebGL 1.0
      int powi = 1;
      if (index == 0) {
        powi = 1;
      } else if (index == 1) {
        powi = 2;
      } else if (index == 2) {
        powi = 4;
      } else if (index == 3) {
        powi = 8;
      } else if (index == 4) {
        powi = 16;
      } else if (index == 5) {
        powi = 32;
      } else if (index == 6) {
        powi = 64;
      } else if (index == 7) {
        powi = 128;
      }
    
      int ndp = number / powi;
    
      return mod(float(ndp), 2.0) != 0.0;
    }
    
    /**
     * Gets the the LOD at the point position.
     */
    float getLOD() {
      vec3 offset = vec3(0.0, 0.0, 0.0);
      int iOffset = int(vnStart);
      float depth = level;
    
      for (float i = 0.0; i <= 30.0; i++) {
        float nodeSizeAtLevel = octreeSize  / pow(2.0, i + level + 0.0);
    
        vec3 index3d = (position-offset) / nodeSizeAtLevel;
        index3d = floor(index3d + 0.5);
        int index = int(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z));
    
        vec4 value = texture2D(visibleNodes, vec2(float(iOffset) / 2048.0, 0.0));
        int mask = int(round(value.r * 255.0));
    
        if (isBitSet(mask, index)) {
          // there are more visible child nodes at this position
          int advanceG = int(round(value.g * 255.0)) * 256;
          int advanceB = int(round(value.b * 255.0));
          int advanceChild = numberOfOnes(mask, index - 1);
          int advance = advanceG + advanceB + advanceChild;
    
          iOffset = iOffset + advance;
    
          depth++;
        } else {
          return value.a * 255.0; // no more visible child nodes at this position
        }
    
        offset = offset + (vec3(1.0, 1.0, 1.0) * nodeSizeAtLevel * 0.5) * index3d;
      }
    
      return depth;
    }
    
    float getPointSizeAttenuation() {
      return 0.5 * pow(2.0, getLOD());
    }
    
    #endif
    
    // ---------------------
    // KD-TREE
    // ---------------------
    
    #if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_kdtree)
    
    float getLOD() {
      vec3 offset = vec3(0.0, 0.0, 0.0);
      float intOffset = 0.0;
      float depth = 0.0;
    
      vec3 size = bbSize;
      vec3 pos = position;
    
      for (float i = 0.0; i <= 1000.0; i++) {
    
        vec4 value = texture2D(visibleNodes, vec2(intOffset / 2048.0, 0.0));
    
        int children = int(value.r * 255.0);
        float next = value.g * 255.0;
        int split = int(value.b * 255.0);
    
        if (next == 0.0) {
           return depth;
        }
    
        vec3 splitv = vec3(0.0, 0.0, 0.0);
        if (split == 1) {
          splitv.x = 1.0;
        } else if (split == 2) {
           splitv.y = 1.0;
        } else if (split == 4) {
           splitv.z = 1.0;
        }
    
        intOffset = intOffset + next;
    
        float factor = length(pos * splitv / size);
        if (factor < 0.5) {
           // left
          if (children == 0 || children == 2) {
            return depth;
          }
        } else {
          // right
          pos = pos - size * splitv * 0.5;
          if (children == 0 || children == 1) {
            return depth;
          }
          if (children == 3) {
            intOffset = intOffset + 1.0;
          }
        }
        size = size * ((1.0 - (splitv + 1.0) / 2.0) + 0.5);
    
        depth++;
      }
    
    
      return depth;
    }
    
    float getPointSizeAttenuation() {
      return 0.5 * pow(1.3, getLOD());
    }
    
    #endif
    
    // formula adapted from: http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    float getContrastFactor(float contrast) {
      return (1.0158730158730156 * (contrast + 1.0)) / (1.0158730158730156 - contrast);
    }
    
    vec3 getRGB() {
      
      #ifdef color_rgba
        vec3 rgb = rgba.rgb;
      #else	
        vec3 rgb = color;
      #endif		
    
      #if defined(use_rgb_gamma_contrast_brightness)
        rgb = pow(rgb, vec3(rgbGamma));
        rgb = rgb + rgbBrightness;
        rgb = (rgb - 0.5) * getContrastFactor(rgbContrast) + 0.5;
        rgb = clamp(rgb, 0.0, 1.0);
        return rgb;
      #else
        return rgb;
      #endif
    }
    
    float getIntensity() {
      float w = (intensity - intensityRange.x) / (intensityRange.y - intensityRange.x);
      w = pow(w, intensityGamma);
      w = w + intensityBrightness;
      w = (w - 0.5) * getContrastFactor(intensityContrast) + 0.5;
      w = clamp(w, 0.0, 1.0);
    
      return w;
    }
    
    vec3 getElevation() {
      vec4 world = modelMatrix * vec4( position, 1.0 );
      float w = (world.z - heightMin) / (heightMax-heightMin);
      vec3 cElevation = texture2D(gradient, vec2(w,1.0-w)).rgb;
    
      return cElevation;
    }
    
    vec4 getClassification() {
      vec2 uv = vec2(classification / 255.0, 0.5);
      vec4 classColor = texture2D(classificationLUT, uv);
    
      return classColor;
    }
    
    vec3 getReturnNumber() {
      if (numberOfReturns == 1.0) {
        return vec3(1.0, 1.0, 0.0);
      } else {
        if (returnNumber == 1.0) {
          return vec3(1.0, 0.0, 0.0);
        } else if (returnNumber == numberOfReturns) {
          return vec3(0.0, 0.0, 1.0);
        } else {
          return vec3(0.0, 1.0, 0.0);
        }
      }
    }
    
    vec3 getSourceID() {
      float w = mod(pointSourceID, 10.0) / 10.0;
      return texture2D(gradient, vec2(w, 1.0 - w)).rgb;
    }
    
    vec3 getCompositeColor() {
      vec3 c;
      float w;
    
      c += wRGB * getRGB();
      w += wRGB;
    
      c += wIntensity * getIntensity() * vec3(1.0, 1.0, 1.0);
      w += wIntensity;
    
      c += wElevation * getElevation();
      w += wElevation;
    
      c += wReturnNumber * getReturnNumber();
      w += wReturnNumber;
    
      c += wSourceID * getSourceID();
      w += wSourceID;
    
      vec4 cl = wClassification * getClassification();
      c += cl.a * cl.rgb;
      w += wClassification * cl.a;
    
      c = c / w;
    
      if (w == 0.0) {
        gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
      }
    
      return c;
    }
    
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    
      gl_Position = projectionMatrix * mvPosition;
    
      #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0) || defined(paraboloid_point_shape)
        vViewPosition = mvPosition.xyz;
      #endif
    
      #if defined weighted_splats
        vLinearDepth = gl_Position.w;
      #endif
    
      #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0)
        vNormal = normalize(normalMatrix * normal);
      #endif
    
      #if !defined(paraboloid_point_shape) && defined(use_edl)
        vLogDepth = log2(-mvPosition.z);
      #endif
    
      // ---------------------
      // POINT SIZE
      // ---------------------
    
      float pointSize = 1.0;
      float slope = tan(fov / 2.0);
      float projFactor =  -0.5 * screenHeight / (slope * mvPosition.z);
    
      #if defined fixed_point_size
        pointSize = size;
      #elif defined attenuated_point_size
        pointSize = size * spacing * projFactor;
      #elif defined adaptive_point_size
        float worldSpaceSize = 2.0 * size * spacing / getPointSizeAttenuation();
        pointSize = worldSpaceSize * projFactor;
      #endif
    
      pointSize = max(minSize, pointSize);
      pointSize = min(maxSize, pointSize);
    
      #if defined(weighted_splats) || defined(paraboloid_point_shape)
        vRadius = pointSize / projFactor;
      #endif
    
      gl_PointSize = pointSize;
    
      // ---------------------
      // HIGHLIGHTING
      // ---------------------
    
      #ifdef highlight_point
        vec4 mPosition = modelMatrix * vec4(position, 1.0);
        if (enablePointHighlighting && abs(mPosition.x - highlightedPointCoordinate.x) < 0.0001 &&
          abs(mPosition.y - highlightedPointCoordinate.y) < 0.0001 &&
          abs(mPosition.z - highlightedPointCoordinate.z) < 0.0001) {
          vHighlight = 1.0;
          gl_PointSize = pointSize * highlightedPointScale;
        } else {
          vHighlight = 0.0;
        }
      #endif
    
      // ---------------------
      // OPACITY
      // ---------------------
    
      #ifndef color_type_point_index
        #ifdef attenuated_opacity
          vOpacity = opacity * exp(-length(-mvPosition.xyz) / opacityAttenuation);
        #else
          vOpacity = opacity;
        #endif
      #endif
    
      // ---------------------
      // FILTERING
      // ---------------------
    
      #ifdef use_filter_by_normal
        bool discardPoint = false;
        // Absolute normal filtering
        if (normalFilteringMode == 1) {
          discardPoint = (abs((modelViewMatrix * vec4(normal, 0.0)).z) > filterByNormalThreshold);
        }
        // less than equal to
        else if (normalFilteringMode == 2) {
          discardPoint = (modelViewMatrix * vec4(normal, 0.0)).z <= filterByNormalThreshold;
          }
        // greater than
        else if(normalFilteringMode == 3) {
          discardPoint = (modelViewMatrix * vec4(normal, 0.0)).z > filterByNormalThreshold;
          }
    
        if (discardPoint)
        {
          gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        }
      #endif
    
      // ---------------------
      // POINT COLOR
      // ---------------------
    
      #ifdef color_type_rgb
        vColor = getRGB();
      #elif defined color_type_height
        vColor = getElevation();
      #elif defined color_type_rgb_height
        vec3 cHeight = getElevation();
        vColor = (1.0 - transition) * getRGB() + transition * cHeight;
      #elif defined color_type_depth
        float linearDepth = -mvPosition.z ;
        float expDepth = (gl_Position.z / gl_Position.w) * 0.5 + 0.5;
        vColor = vec3(linearDepth, expDepth, 0.0);
      #elif defined color_type_intensity
        float w = getIntensity();
        vColor = vec3(w, w, w);
      #elif defined color_type_intensity_gradient
        float w = getIntensity();
        vColor = texture2D(gradient, vec2(w, 1.0 - w)).rgb;
      #elif defined color_type_color
        vColor = uColor;
      #elif defined color_type_lod
      float w = getLOD() / 10.0;
      vColor = texture2D(gradient, vec2(w, 1.0 - w)).rgb;
      #elif defined color_type_point_index
        vColor = indices.rgb;
      #elif defined color_type_classification
        vec4 cl = getClassification();
        vColor = cl.rgb;
      #elif defined color_type_return_number
        vColor = getReturnNumber();
      #elif defined color_type_source
        vColor = getSourceID();
      #elif defined color_type_normal
        vColor = (modelMatrix * vec4(normal, 0.0)).xyz;
      #elif defined color_type_phong
        vColor = color;
      #elif defined color_type_composite
        vColor = getCompositeColor();
      #endif
    
      #if !defined color_type_composite && defined color_type_classification
        if (cl.a == 0.0) {
          gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
          return;
        }
      #endif
    
      // ---------------------
      // CLIPPING
      // ---------------------
    
      #if defined use_clip_box
        bool insideAny = false;
        for (int i = 0; i < max_clip_boxes; i++) {
          if (i == int(clipBoxCount)) {
            break;
          }
    
          vec4 clipPosition = clipBoxes[i] * modelMatrix * vec4(position, 1.0);
          bool inside = -0.5 <= clipPosition.x && clipPosition.x <= 0.5;
          inside = inside && -0.5 <= clipPosition.y && clipPosition.y <= 0.5;
          inside = inside && -0.5 <= clipPosition.z && clipPosition.z <= 0.5;
          insideAny = insideAny || inside;
        }
    
        if (!insideAny) {
          #if defined clip_outside
            gl_Position = vec4(1000.0, 1000.0, 1000.0, 1.0);
          #elif defined clip_highlight_inside && !defined(color_type_depth)
            float c = (vColor.r + vColor.g + vColor.b) / 6.0;
          #endif
        } else {
          #if defined clip_highlight_inside
            vColor.r += 0.5;
          #endif
        }
      #endif
    }
    `;
    const fragSource = `precision highp float;
    precision highp int;
    
    #if defined paraboloid_point_shape
      #extension GL_EXT_frag_depth : enable
    #endif
    
    uniform mat4 viewMatrix;
    uniform vec3 cameraPosition;
    
    uniform mat4 projectionMatrix;
    uniform float opacity;
    
    uniform float blendHardness;
    uniform float blendDepthSupplement;
    uniform float fov;
    uniform float spacing;
    uniform float pcIndex;
    uniform float screenWidth;
    uniform float screenHeight;
    
    uniform sampler2D depthMap;
    
    #if defined (clip_horizontally) || defined (clip_vertically)
      uniform vec4 clipExtent;
    #endif
    
    #ifdef use_texture_blending
      uniform sampler2D backgroundMap;
    #endif
    
    
    #ifdef use_point_cloud_mixing
      uniform int pointCloudMixingMode;
      uniform float pointCloudID;
      uniform float pointCloudMixAngle;
      uniform float stripeDistanceX;
      uniform float stripeDistanceY;
    
      uniform float stripeDivisorX;
      uniform float stripeDivisorY;
    #endif
    
    #ifdef highlight_point
      uniform vec4 highlightedPointColor;
    #endif
    
    varying vec3 vColor;
    
    #if !defined(color_type_point_index)
      varying float vOpacity;
    #endif
    
    #if defined(weighted_splats)
      varying float vLinearDepth;
    #endif
    
    #if !defined(paraboloid_point_shape) && defined(use_edl)
      varying float vLogDepth;
    #endif
    
    #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0) || defined(paraboloid_point_shape)
      varying vec3 vViewPosition;
    #endif
    
    #if defined(weighted_splats) || defined(paraboloid_point_shape)
      varying float vRadius;
    #endif
    
    #if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0)
      varying vec3 vNormal;
    #endif
    
    #ifdef highlight_point
      varying float vHighlight;
    #endif
    
    float specularStrength = 1.0;
    
    void main() {
      vec3 color = vColor;
      float depth = gl_FragCoord.z;
    
      #if defined (clip_horizontally) || defined (clip_vertically)
      vec2 ndc = vec2((gl_FragCoord.x / screenWidth), 1.0 - (gl_FragCoord.y / screenHeight));
    
      if(step(clipExtent.x, ndc.x) * step(ndc.x, clipExtent.z) < 1.0)
      {
        discard;
      }
    
      if(step(clipExtent.y, ndc.y) * step(ndc.y, clipExtent.w) < 1.0)
      {
        discard;
      }
      #endif  
    
      #if defined(circle_point_shape) || defined(paraboloid_point_shape) || defined (weighted_splats)
        float u = 2.0 * gl_PointCoord.x - 1.0;
        float v = 2.0 * gl_PointCoord.y - 1.0;
      #endif
    
      #if defined(circle_point_shape) || defined (weighted_splats)
        float cc = u*u + v*v;
        if(cc > 1.0){
          discard;
        }
      #endif
    
      #if defined weighted_splats
        vec2 uv = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
        float sDepth = texture2D(depthMap, uv).r;
        if(vLinearDepth > sDepth + vRadius + blendDepthSupplement){
          discard;
        }
      #endif
    
      #if defined color_type_point_index
        gl_FragColor = vec4(color, pcIndex / 255.0);
      #else
        gl_FragColor = vec4(color, vOpacity);
      #endif
    
      #ifdef use_point_cloud_mixing
        bool discardFragment = false;
    
        if (pointCloudMixingMode == 1) {  // Checkboard
          float vPointCloudID = pointCloudID > 10. ? pointCloudID/10.: pointCloudID;
          discardFragment = mod(gl_FragCoord.x, vPointCloudID) > 0.5 && mod(gl_FragCoord.y, vPointCloudID) > 0.5;
        }
        else if (pointCloudMixingMode == 2) {  // Stripes
          float angle = pointCloudMixAngle * pointCloudID / 180.;
          float u = cos(angle) * gl_FragCoord.x + sin(angle) * gl_FragCoord.y;
          float v = -sin(angle) * gl_FragCoord.x + cos(angle) * gl_FragCoord.y;
    
          discardFragment = mod(u, stripeDistanceX) >= stripeDistanceX/stripeDivisorX && mod(v, stripeDistanceY) >= stripeDistanceY/stripeDivisorY;
        }
        if (discardFragment) {
          discard;
        }
      #endif
    
      #ifdef use_texture_blending
        vec2 vUv = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
    
        vec4 tColor = texture2D(backgroundMap, vUv);
        gl_FragColor = vec4(vOpacity * color, 1.) + vec4((1. - vOpacity) * tColor.rgb, 0.);
      #endif
    
      #if defined(color_type_phong)
        #if MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0
          vec3 normal = normalize( vNormal );
          normal.z = abs(normal.z);
    
          vec3 viewPosition = normalize( vViewPosition );
        #endif
    
        // code taken from three.js phong light fragment shader
    
        #if MAX_POINT_LIGHTS > 0
    
          vec3 pointDiffuse = vec3( 0.0 );
          vec3 pointSpecular = vec3( 0.0 );
    
          for ( int i = 0; i < MAX_POINT_LIGHTS; i ++ ) {
    
            vec4 lPosition = viewMatrix * vec4( pointLightPosition[ i ], 1.0 );
            vec3 lVector = lPosition.xyz + vViewPosition.xyz;
    
            float lDistance = 1.0;
            if ( pointLightDistance[ i ] > 0.0 )
              lDistance = 1.0 - min( ( length( lVector ) / pointLightDistance[ i ] ), 1.0 );
    
            lVector = normalize( lVector );
    
                // diffuse
    
            float dotProduct = dot( normal, lVector );
    
            #ifdef WRAP_AROUND
    
              float pointDiffuseWeightFull = max( dotProduct, 0.0 );
              float pointDiffuseWeightHalf = max( 0.5 * dotProduct + 0.5, 0.0 );
    
              vec3 pointDiffuseWeight = mix( vec3( pointDiffuseWeightFull ), vec3( pointDiffuseWeightHalf ), wrapRGB );
    
            #else
    
              float pointDiffuseWeight = max( dotProduct, 0.0 );
    
            #endif
    
            pointDiffuse += diffuse * pointLightColor[ i ] * pointDiffuseWeight * lDistance;
    
            // specular
    
            vec3 pointHalfVector = normalize( lVector + viewPosition );
            float pointDotNormalHalf = max( dot( normal, pointHalfVector ), 0.0 );
            float pointSpecularWeight = specularStrength * max( pow( pointDotNormalHalf, shininess ), 0.0 );
    
            float specularNormalization = ( shininess + 2.0 ) / 8.0;
    
            vec3 schlick = specular + vec3( 1.0 - specular ) * pow( max( 1.0 - dot( lVector, pointHalfVector ), 0.0 ), 5.0 );
            pointSpecular += schlick * pointLightColor[ i ] * pointSpecularWeight * pointDiffuseWeight * lDistance * specularNormalization;
            pointSpecular = vec3(0.0, 0.0, 0.0);
          }
    
        #endif
    
        #if MAX_DIR_LIGHTS > 0
    
          vec3 dirDiffuse = vec3( 0.0 );
          vec3 dirSpecular = vec3( 0.0 );
    
          for( int i = 0; i < MAX_DIR_LIGHTS; i ++ ) {
    
            vec4 lDirection = viewMatrix * vec4( directionalLightDirection[ i ], 0.0 );
            vec3 dirVector = normalize( lDirection.xyz );
    
                // diffuse
    
            float dotProduct = dot( normal, dirVector );
    
            #ifdef WRAP_AROUND
    
              float dirDiffuseWeightFull = max( dotProduct, 0.0 );
              float dirDiffuseWeightHalf = max( 0.5 * dotProduct + 0.5, 0.0 );
    
              vec3 dirDiffuseWeight = mix( vec3( dirDiffuseWeightFull ), vec3( dirDiffuseWeightHalf ), wrapRGB );
    
            #else
    
              float dirDiffuseWeight = max( dotProduct, 0.0 );
    
            #endif
    
            dirDiffuse += diffuse * directionalLightColor[ i ] * dirDiffuseWeight;
    
            // specular
    
            vec3 dirHalfVector = normalize( dirVector + viewPosition );
            float dirDotNormalHalf = max( dot( normal, dirHalfVector ), 0.0 );
            float dirSpecularWeight = specularStrength * max( pow( dirDotNormalHalf, shininess ), 0.0 );
    
            float specularNormalization = ( shininess + 2.0 ) / 8.0;
    
            vec3 schlick = specular + vec3( 1.0 - specular ) * pow( max( 1.0 - dot( dirVector, dirHalfVector ), 0.0 ), 5.0 );
            dirSpecular += schlick * directionalLightColor[ i ] * dirSpecularWeight * dirDiffuseWeight * specularNormalization;
          }
    
        #endif
    
        vec3 totalDiffuse = vec3( 0.0 );
        vec3 totalSpecular = vec3( 0.0 );
    
        #if MAX_POINT_LIGHTS > 0
    
          totalDiffuse += pointDiffuse;
          totalSpecular += pointSpecular;
    
        #endif
    
        #if MAX_DIR_LIGHTS > 0
    
          totalDiffuse += dirDiffuse;
          totalSpecular += dirSpecular;
    
        #endif
    
        gl_FragColor.xyz = gl_FragColor.xyz * ( emissive + totalDiffuse + ambientLightColor * ambient ) + totalSpecular;
    
      #endif
    
      #if defined weighted_splats
          //float w = pow(1.0 - (u*u + v*v), blendHardness);
    
        float wx = 2.0 * length(2.0 * gl_PointCoord - 1.0);
        float w = exp(-wx * wx * 0.5);
    
        //float distance = length(2.0 * gl_PointCoord - 1.0);
        //float w = exp( -(distance * distance) / blendHardness);
    
        gl_FragColor.rgb = gl_FragColor.rgb * w;
        gl_FragColor.a = w;
      #endif
    
      #if defined paraboloid_point_shape
        float wi = 0.0 - ( u*u + v*v);
        vec4 pos = vec4(vViewPosition, 1.0);
        pos.z += wi * vRadius;
        float linearDepth = -pos.z;
        pos = projectionMatrix * pos;
        pos = pos / pos.w;
        float expDepth = pos.z;
        depth = (pos.z + 1.0) / 2.0;
        gl_FragDepthEXT = depth;
    
        #if defined(color_type_depth)
          gl_FragColor.r = linearDepth;
          gl_FragColor.g = expDepth;
        #endif
    
        #if defined(use_edl)
          gl_FragColor.a = log2(linearDepth);
        #endif
    
      #else
        #if defined(use_edl)
          gl_FragColor.a = vLogDepth;
        #endif
      #endif
    
      #ifdef highlight_point
        if (vHighlight > 0.0) {
          gl_FragColor = highlightedPointColor;
        }
      #endif
    }
    `;
    this.vertexShader = this.applyDefines(vertSource);
    this.fragmentShader = this.applyDefines(fragSource);
   // this.vertexShader = this.applyDefines(require('./shaders/pointcloud.vert').default);
   // this.fragmentShader = this.applyDefines(require('./shaders/pointcloud.frag').default);

    if (this.opacity === 1.0) {
      this.blending = NoBlending;
      this.transparent = false;
      this.depthTest = true;
      this.depthWrite = true;
      this.depthFunc = LessEqualDepth;
    } else if (this.opacity < 1.0 && !this.useEDL) {
      this.blending = AdditiveBlending;
      this.transparent = true;
      this.depthTest = false;
      this.depthWrite = true;
    }

    if (this.weighted) {
      this.blending = AdditiveBlending;
      this.transparent = true;
      this.depthTest = true;
      this.depthWrite = false;
      this.depthFunc = LessEqualDepth;
    }

    this.needsUpdate = true;
  }

  applyDefines(shaderSrc: string): string {
    const parts: string[] = [];

    function define(value: string | undefined) {
      if (value) {
        parts.push(`#define ${value}`);
      }
    }

    define(TREE_TYPE_DEFS[this.treeType]);
    define(SIZE_TYPE_DEFS[this.pointSizeType]);
    define(SHAPE_DEFS[this.shape]);
    define(COLOR_DEFS[this.pointColorType]);
    define(CLIP_MODE_DEFS[this.clipMode]);
    define(OPACITY_DEFS[this.pointOpacityType]);

    // We only perform gamma and brightness/contrast calculations per point if values are specified.
    if (
      this.rgbGamma !== DEFAULT_RGB_GAMMA ||
      this.rgbBrightness !== DEFAULT_RGB_BRIGHTNESS ||
      this.rgbContrast !== DEFAULT_RGB_CONTRAST
    ) {
      define('use_rgb_gamma_contrast_brightness');
    }

    if (this.useFilterByNormal) {
      define('use_filter_by_normal');
    }

    if (this.useEDL) {
      define('use_edl');
    }

    if (this.weighted) {
      define('weighted_splats');
    }

    if (this.numClipBoxes > 0) {
      define('use_clip_box');
    }

    if (this.highlightPoint) {
      define('highlight_point');
    }

    if (this.useTextureBlending) {
      define('use_texture_blending');
    }

    if (this.usePointCloudMixing) {
      define('use_point_cloud_mixing');
    }

    if (this.colorRgba) {
      define('color_rgba');
    }

    define('MAX_POINT_LIGHTS 0');
    define('MAX_DIR_LIGHTS 0');

    parts.push(shaderSrc);

    return parts.join('\n');
  }

  setPointCloudMixingMode(mode: PointCloudMixingMode) {
    this.pointCloudMixingMode = mode;
  }

  getPointCloudMixingMode(): PointCloudMixingMode {
    if (this.pointCloudMixingMode === PointCloudMixingMode.STRIPES) {
      return PointCloudMixingMode.STRIPES;
    }

    return PointCloudMixingMode.CHECKBOARD;
  }

  setClipBoxes(clipBoxes: IClipBox[]): void {
    if (!clipBoxes) {
      return;
    }

    this.clipBoxes = clipBoxes;

    const doUpdate =
      this.numClipBoxes !== clipBoxes.length && (clipBoxes.length === 0 || this.numClipBoxes === 0);

    this.numClipBoxes = clipBoxes.length;
    this.setUniform('clipBoxCount', this.numClipBoxes);

    if (doUpdate) {
      this.updateShaderSource();
    }

    const clipBoxesLength = this.numClipBoxes * 16;
    const clipBoxesArray = new Float32Array(clipBoxesLength);

    for (let i = 0; i < this.numClipBoxes; i++) {
      clipBoxesArray.set(clipBoxes[i].inverse.elements, 16 * i);
    }

    for (let i = 0; i < clipBoxesLength; i++) {
      if (isNaN(clipBoxesArray[i])) {
        clipBoxesArray[i] = Infinity;
      }
    }

    this.setUniform('clipBoxes', clipBoxesArray);
  }

  get gradient(): IGradient {
    return this._gradient;
  }

  set gradient(value: IGradient) {
    if (this._gradient !== value) {
      this._gradient = value;
      this.gradientTexture = generateGradientTexture(this._gradient);
      this.setUniform('gradient', this.gradientTexture);
    }
  }

  get classification(): IClassification {
    return this._classification;
  }

  set classification(value: IClassification) {
    const copy: IClassification = {} as any;
    for (const key of Object.keys(value)) {
      copy[key] = value[key].clone();
    }

    let isEqual = false;
    if (this._classification === undefined) {
      isEqual = false;
    } else {
      isEqual = Object.keys(copy).length === Object.keys(this._classification).length;

      for (const key of Object.keys(copy)) {
        isEqual = isEqual && this._classification[key] !== undefined;
        isEqual = isEqual && copy[key].equals(this._classification[key]);
      }
    }

    if (!isEqual) {
      this._classification = copy;
      this.recomputeClassification();
    }
  }

  private recomputeClassification(): void {
    this.classificationTexture = generateClassificationTexture(this._classification);
    this.setUniform('classificationLUT', this.classificationTexture);
  }

  get elevationRange(): [number, number] {
    return [this.heightMin, this.heightMax];
  }

  set elevationRange(value: [number, number]) {
    this.heightMin = value[0];
    this.heightMax = value[1];
  }

  getUniform<K extends keyof IPointCloudMaterialUniforms>(
    name: K,
  ): IPointCloudMaterialUniforms[K]['value'] {
    return this.uniforms === undefined ? (undefined as any) : this.uniforms[name].value;
  }

  setUniform<K extends keyof IPointCloudMaterialUniforms>(
    name: K,
    value: IPointCloudMaterialUniforms[K]['value'],
  ): void {
    if (this.uniforms === undefined) {
      return;
    }

    const uObj = this.uniforms[name];

    if (uObj.type === 'c') {
      (uObj.value as Color).copy(value as Color);
    } else if (value !== uObj.value) {
      uObj.value = value;
    }
  }

  updateMaterial(
    octree: PointCloudOctree,
    visibleNodes: PointCloudOctreeNode[],
    camera: Camera,
    renderer: WebGLRenderer,
  ): void {
    const pixelRatio = renderer.getPixelRatio();

    if (camera.type === PERSPECTIVE_CAMERA) {
      this.fov = (camera as PerspectiveCamera).fov * (Math.PI / 180);
    } else {
      this.fov = Math.PI / 2; // will result in slope = 1 in the shader
    }
    const renderTarget = renderer.getRenderTarget();
    if (renderTarget !== null) {
      this.screenWidth = renderTarget.width;
      this.screenHeight = renderTarget.height;
    } else {
      this.screenWidth = renderer.domElement.clientWidth * pixelRatio;
      this.screenHeight = renderer.domElement.clientHeight * pixelRatio;
    }

    if (this.useDrawingBufferSize) {
      renderer.getDrawingBufferSize(PointCloudMaterial.helperVec2);
      this.screenWidth = PointCloudMaterial.helperVec2.width;
      this.screenHeight = PointCloudMaterial.helperVec2.height;
    }

    const maxScale = Math.max(octree.scale.x, octree.scale.y, octree.scale.z);
    this.spacing = octree.pcoGeometry.spacing * maxScale;
    this.octreeSize = octree.pcoGeometry.boundingBox.getSize(PointCloudMaterial.helperVec3).x;

    if (
      this.pointSizeType === PointSizeType.ADAPTIVE ||
      this.pointColorType === PointColorType.LOD
    ) {
      this.updateVisibilityTextureData(visibleNodes);
    }
  }

  private updateVisibilityTextureData(nodes: PointCloudOctreeNode[]) {
    nodes.sort(byLevelAndIndex);

    const data = new Uint8Array(nodes.length * 4);
    const offsetsToChild = new Array(nodes.length).fill(Infinity);

    this.visibleNodeTextureOffsets.clear();

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];

      this.visibleNodeTextureOffsets.set(node.name, i);

      if (i > 0) {
        const parentName = node.name.slice(0, -1);
        const parentOffset = this.visibleNodeTextureOffsets.get(parentName)!;
        const parentOffsetToChild = i - parentOffset;

        offsetsToChild[parentOffset] = Math.min(offsetsToChild[parentOffset], parentOffsetToChild);

        // tslint:disable:no-bitwise
        const offset = parentOffset * 4;
        data[offset] = data[offset] | (1 << node.index);
        data[offset + 1] = offsetsToChild[parentOffset] >> 8;
        data[offset + 2] = offsetsToChild[parentOffset] % 256;
        // tslint:enable:no-bitwise
      }

      data[i * 4 + 3] = node.name.length;
    }

    const texture = this.visibleNodesTexture;
    if (texture) {
      texture.image.data.set(data);
      texture.needsUpdate = true;
    }
  }

  static makeOnBeforeRender(
    octree: PointCloudOctree,
    node: PointCloudOctreeNode,
    pcIndex?: number,
  ) {
    return (
      _renderer: WebGLRenderer,
      _scene: Scene,
      _camera: Camera,
      _geometry: BufferGeometry,
      material: Material,
    ) => {
      const pointCloudMaterial = material as PointCloudMaterial;
      const materialUniforms = pointCloudMaterial.uniforms;

      materialUniforms.level.value = node.level;
      materialUniforms.isLeafNode.value = node.isLeafNode;

      const vnStart = pointCloudMaterial.visibleNodeTextureOffsets.get(node.name);
      if (vnStart !== undefined) {
        materialUniforms.vnStart.value = vnStart;
      }

      materialUniforms.pcIndex.value =
        pcIndex !== undefined ? pcIndex : octree.visibleNodes.indexOf(node);

      // Note: when changing uniforms in onBeforeRender, the flag uniformsNeedUpdate has to be
      // set to true to instruct ThreeJS to upload them. See also
      // https://github.com/mrdoob/three.js/issues/9870#issuecomment-368750182.

      // Remove the cast to any after updating to Three.JS >= r113
      (material as any) /*ShaderMaterial*/.uniformsNeedUpdate = true;
    };
  }
}

function makeUniform<T>(type: string, value: T): IUniform<T> {
  return { type, value };
}

function getValid<T>(a: T | undefined, b: T): T {
  return a === undefined ? b : a;
}

// tslint:disable:no-invalid-this
function uniform<K extends keyof IPointCloudMaterialUniforms>(
  uniformName: K,
  requireSrcUpdate: boolean = false,
): PropertyDecorator {
  return (target: Object, propertyKey: string | symbol): void => {
    Object.defineProperty(target, propertyKey, {
      get() {
        return this.getUniform(uniformName);
      },
      set(value: any) {
        if (value !== this.getUniform(uniformName)) {
          this.setUniform(uniformName, value);
          if (requireSrcUpdate) {
            this.updateShaderSource();
          }
        }
      },
    });
  };
}

function requiresShaderUpdate() {
  return (target: Object, propertyKey: string | symbol): void => {
    const fieldName = `_${propertyKey.toString()}`;

    Object.defineProperty(target, propertyKey, {
      get() {
        return this[fieldName];
      },
      set(value: any) {
        if (value !== this[fieldName]) {
          this[fieldName] = value;
          this.updateShaderSource();
        }
      },
    });
  };
}
