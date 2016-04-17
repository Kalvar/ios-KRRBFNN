//
//  KRRBFNN.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "KRRBFHiddenLayer.h"
#import "KRRBFOutputLayer.h"

#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"

#import "KRRBFOLS.h"
#import "KRRBFRandom.h"
#import "KRRBFLMS.h"
#import "KRRBFSGA.h"

@class KRRBFTarget;
@class KRRBFNN;

typedef void(^KRRBFNNCompletion)(BOOL success, KRRBFNN *rbfnn);
typedef void(^KRRBFNNPatternOutput)(NSArray <KRRBFOutputNet *> *patternOutputs);
typedef BOOL(^KRRBFNNIteration)(NSInteger iteration, double rmse, NSArray <KRRBFCenterNet *> *centers, NSArray <KRRBFOutputNet *> *weights, double sigma); // Returns NO that means immediately stop iteration calculation.
typedef void(^KRRBFNNPredication)(NSDictionary <NSString *, NSArray <NSNumber *> *> *outputs);

@interface KRRBFNN : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *patterns;
@property (nonatomic, strong) NSMutableArray <KRRBFTarget *> *targets;

@property (nonatomic, readonly) NSMutableArray <KRRBFCenterNet *> *centers;
@property (nonatomic, readonly) NSMutableArray <KRRBFOutputNet *> *weights; // Network weights are included in "KROutputLayer.nets".

@property (nonatomic, strong) KRRBFHiddenLayer *hiddenLayer;
@property (nonatomic, strong) KRRBFOutputLayer *outputLayer;

@property (nonatomic, readonly) double rmse;              // Current network RMSE.
//@property (nonatomic, readonly) NSArray *sigmas;          // Current centers of network that sigma values, if we used SGA to learn that weights of network, centers are own their specific sigma value, it means 1 center has 1 sigma.
@property (nonatomic, assign) NSInteger maxIteration;     // Max iteration to limit network training times.
@property (nonatomic, assign) double toleranceError;      // Tolerance error of RMSE value to use on SGA judges when to stop.
@property (nonatomic, assign) double learningRate;        // Learning rate for SGA.

+(instancetype)sharedNetwork;
-(instancetype)init;

-(void)recoverForKey:(NSString *)_key; // Recovering centers and weights from saved network information.
-(void)removeForKey:(NSString *)_key;  // Removes saved network information.
-(void)saveForKey:(NSString *)_key;    // Saving current trained network information.
-(void)reset;                          // Cleans all trained information
-(void)removeCachesIfNeeded;

-(void)addPattern:(KRRBFPattern *)_pattern;
-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples;

-(void)randomWeightsBetweenMin:(double)_minValue max:(double)_maxValue;
-(void)addCenters:(NSArray <KRRBFCenterNet *> *)_objects;
-(void)addWeights:(NSArray <KRRBFOutputNet *> *)_objects;

-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance toSave:(BOOL)_toSave;
-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance;
-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount toSave:(BOOL)_toSave;
-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount;

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion patternOutput:(KRRBFNNPatternOutput)_patternOutput; // patternOutput is that outpus of each pattern.
-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion;

-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion iteration:(KRRBFNNIteration)_iteration;
-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion;

-(void)predicateWithPatterns:(NSArray <KRRBFPattern *> *)_predicatePatterns output:(KRRBFNNPredication)_outputsBlock;

@end
