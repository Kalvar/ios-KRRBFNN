//
//  KRRBFActiviation.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFActiviation : NSObject

+(instancetype)sharedActiviation;
-(instancetype)init;

-(double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2;
-(double)rbf:(NSArray *)_x1 x2:(NSArray *)_x2 sigma:(float)_sigma;

@end
