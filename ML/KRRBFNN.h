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

typedef void(^KRRBFNNCompletion)(BOOL success, KRRBFNN *rbfnn, double rmse);
typedef void(^KRRBFNNEachOutput)(KRRBFOutputNet *outputNet);
typedef void(^KRRBFNNPredication)(NSDictionary <NSString *, NSArray <NSNumber *> *> *outputs);

@interface KRRBFNN : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *patterns;
@property (nonatomic, strong) NSMutableArray <KRRBFTarget *> *targets;

@property (nonatomic, readonly) NSMutableArray <KRRBFCenterNet *> *centers;
@property (nonatomic, readonly) NSMutableArray <KRRBFOutputNet *> *weights; // Network weights included in output nets

@property (nonatomic, strong) KRRBFHiddenLayer *hiddenLayer;
@property (nonatomic, strong) KRRBFOutputLayer *outputLayer;

// Center choice methods
@property (nonatomic, strong) KRRBFOLS *ols;
@property (nonatomic, strong) KRRBFRandom *random;

// Learning methods
@property (nonatomic, strong) KRRBFLMS *lms;
@property (nonatomic, strong) KRRBFSGA *sga;

// Current network RMSE
@property (nonatomic, readonly) double rmse;

+(instancetype)sharedNetwork;
-(instancetype)init;

-(void)recoverForKey:(NSString *)_key; // Recovering centers and weights from saved network information.
-(void)removeForKey:(NSString *)_key;  // Removes saved network information.
-(void)saveForKey:(NSString *)_key;    // Saving current trained network information.
-(void)reset;                          // Cleans all trained information

-(void)addPattern:(KRRBFPattern *)_pattern;
-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples;

-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance toSave:(BOOL)_toSave;
-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance;
-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount toSave:(BOOL)_toSave;
-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount;

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion eachOutput:(KRRBFNNEachOutput)_eachOutput;
-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion;
-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion;

-(void)predicateWithPatterns:(NSArray <KRRBFPattern *> *)_predicatePatterns output:(KRRBFNNPredication)_outputsBlock;

@end
