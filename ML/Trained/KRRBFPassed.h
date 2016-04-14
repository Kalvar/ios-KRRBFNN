//
//  KRRBFPassed.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/12.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFCenterNet;
@class KRRBFOutputNet;

// This saved the trained network.
@interface KRRBFPassed : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray <KRRBFCenterNet *> *centers;
@property (nonatomic, strong) NSMutableArray <KRRBFOutputNet *> *weights;

-(instancetype)init;

@end
