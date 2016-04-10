//
//  KRRBFHiddenLayer.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFCenterNet;

@interface KRRBFHiddenLayer : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFCenterNet *> *nets; // Net is center

+(instancetype)sharedLayer;
-(instancetype)init;

-(void)removeAllCenters;
-(void)addCentersFromArray:(NSArray <KRRBFCenterNet *> *)choseCenters;

@end
