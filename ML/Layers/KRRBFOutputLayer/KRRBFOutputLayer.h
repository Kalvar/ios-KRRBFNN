//
//  KRRBFOutputLayer.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFPattern;
@class KRRBFCenterNet;
@class KRRBFOutputNet;
@class KRRBFOutputLayer;

typedef void(^KRRBFOutputLayerPatternOutput)(NSArray <KRRBFOutputNet *> *patternOutputs, double costError);
typedef void(^KRRBFOutputLayerCompletion)(KRRBFOutputLayer *layer);
typedef void(^KRRBFOutputLayerPredication)(NSDictionary <NSString *, NSArray <NSNumber *> *> *predications);

@interface KRRBFOutputLayer : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFOutputNet *> *nets; // Net is output
@property (nonatomic, assign) double rmse;
@property (nonatomic, assign) double costError; // Cost function of all patterns.
//@property (nonatomic, assign) NSInteger iteration;

+(instancetype)sharedLayer;
-(instancetype)init;

-(void)removeAllNets;
-(void)addNetsFromArray:(NSArray<KRRBFOutputNet *> *)_outputNets;

-(void)setupCommonSigmaWithCenters:(NSArray<KRRBFCenterNet *> *)_centers;
-(void)setupSigmasWithCenters:(NSArray<KRRBFCenterNet *> *)_centers matchSigmas:(NSArray <NSNumber *> *)_matchSigmas;

-(void)outputWithPatterns:(NSArray <KRRBFPattern *> *)_patterns centers:(NSArray <KRRBFCenterNet *> *)_centers completion:(KRRBFOutputLayerCompletion)_completion patternOutput:(KRRBFOutputLayerPatternOutput)_patternOutput;

-(void)predicateWithPatterns:(NSArray<KRRBFPattern *> *)_patterns centers:(NSArray<KRRBFCenterNet *> *)_centers outputs:(KRRBFOutputLayerPredication)_outputsBlock;

@end
