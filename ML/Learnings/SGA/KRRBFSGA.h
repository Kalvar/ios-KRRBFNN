//
//  KRRBFSga.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFCenterNet;
@class KRRBFOutputNet;
@class KRRBFPattern;

typedef void(^KRRBFSGACompletion)(BOOL success, double rmse, NSArray *weights, NSArray *centers, double sigma);
typedef BOOL(^KRRBFSGAIteration)(NSInteger iterationTimes, double iterationRMSE, NSArray *weights, NSArray *centers, double sigma);

@class KRRBFOutputNet;

@interface KRRBFSGA : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFCenterNet *> *centers; // Reference with outside centers to direct updated outside parameters.
@property (nonatomic, strong) NSMutableArray <KRRBFOutputNet *> *weights; // Reference with outside weights.
@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *patterns;  // Reference with outside patterns.
@property (nonatomic, assign) double sigma;

@property (nonatomic, assign) double weightLearningRate; // Learning rate of weights updating.
@property (nonatomic, assign) double centerLearningRate; // Learning rate of centers updating.
@property (nonatomic, assign) double sigmaLearningRate;  // Learning rate of sigma updating.
@property (nonatomic, assign) double learningRate;       // If set up this parameter will set all learning rates are same.

@property (nonatomic, assign) double costError;          // Error value use in updating weights, centers and sigma.

+(instancetype)sharedSGA;
-(instancetype)init;

-(NSArray <KRRBFOutputNet *> *)randomWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount betweenMin:(double)_minValue max:(double)_maxValue;
-(NSArray <KRRBFOutputNet *> *)zeroWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount;

-(void)updateWeights;
-(void)updateCenters;
-(void)updateSigmas;

-(void)freeReferences;

@end
