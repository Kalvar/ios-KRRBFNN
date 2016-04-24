//
//  KRRBFOutput.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNet.h"

// Output net
@interface KRRBFOutputNet : KRRBFNet <NSCoding>

@property (nonatomic, strong) NSMutableArray <NSNumber *> *weights; // 連接自己的所有權重值
@property (nonatomic, assign) double outputValue;                   // 網路輸出值，因為是做線性組合，故網路輸出值即為外部 Hidden Layer Nets 的 sum(z(j) * wj) + bias
@property (nonatomic, assign) double targetValue;                   // 期望輸出值
@property (nonatomic, readonly) double outputError;                 // 輸出誤差值
@property (nonatomic, readonly) double costError;                   // 用來修正權重, 中心點, Sigma 的 Cost Function Value

//@property (nonatomic, assign) double bias;             // 偏權值，暫時不處理

-(instancetype)init;

-(void)removeAllWeights;
-(void)addWeightsFromArray:(NSArray *)_outputWeights;
-(void)addWeight:(NSNumber *)_weight;

-(void)outputWithRBFValues:(NSArray *)_rbfValues;

@end
