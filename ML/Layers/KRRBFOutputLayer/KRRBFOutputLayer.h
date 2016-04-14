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

typedef void(^KRRBFOutputLayerOutput)(KRRBFOutputNet *outputNet);
typedef void(^KRRBFOutputLayerCompletion)(double rmse);
typedef void(^KRRBFOutputLayerPredication)(NSDictionary <NSString *, NSArray <NSNumber *> *> *predications);

@interface KRRBFOutputLayer : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFOutputNet *> *nets; // Net is output

+(instancetype)sharedLayer;
-(instancetype)init;

-(void)removeAllNets;
-(void)addNetsFromArray:(NSArray<KRRBFOutputNet *> *)_outputNets;
-(void)outputWithPatterns:(NSArray <KRRBFPattern *> *)_patterns centers:(NSArray <KRRBFCenterNet *> *)_centers completion:(KRRBFOutputLayerCompletion)_completion eachOutput:(KRRBFOutputLayerOutput)_eachOutput;
-(void)predicateWithPatterns:(NSArray<KRRBFPattern *> *)_patterns centers:(NSArray<KRRBFCenterNet *> *)_centers outputs:(KRRBFOutputLayerPredication)_outputsBlock;

@end
