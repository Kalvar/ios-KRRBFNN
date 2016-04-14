//
//  KRRBFNet.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFNet : NSObject <NSCopying, NSCoding>

@property (nonatomic, strong) NSMutableArray *features; // 特幑向量
@property (nonatomic, strong) NSNumber *indexKey;       // 第幾筆資料或第幾顆神經元
@property (nonatomic, assign) double bias;              // 偏權值, default is nothing

@property (nonatomic, weak) NSCoder *coder;             // For NSCoding usage and child-class use in inherited this class.

-(instancetype)init;
-(void)addFeatures:(NSArray *)_f;

@end

@interface KRRBFNet (NSCoding)

-(void)encodeObject:(id)_object forKey:(NSString *)_key;
-(id)decodeForKey:(NSString *)_key;

@end