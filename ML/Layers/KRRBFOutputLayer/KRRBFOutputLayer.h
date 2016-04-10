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

@interface KRRBFOutputLayer : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFOutputNet *> *nets; // Net is output

+(instancetype)sharedLayer;
-(instancetype)init;

-(void)removeAllNets;
-(void)addNetsFromArray:(NSArray<KRRBFOutputNet *> *)_outputNets;
-(void)outputWithPatterns:(NSArray <KRRBFPattern *> *)_patterns centers:(NSArray <KRRBFCenterNet *> *)_centers eachOutput:(KRRBFOutputLayerOutput)_eachOutput completion:(KRRBFOutputLayerCompletion)_completion;

@end
