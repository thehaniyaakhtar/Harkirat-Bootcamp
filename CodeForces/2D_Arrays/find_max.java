// find maximum
import java.io.*;
import java.util.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;
        
        String[] firstLine = br.readLine().split(" ");
        int n = Integer.parseInt(firstLine[0]);
        int m = Integer.parseInt(firstLine[1]);
        
        int max = Integer.MIN_VALUE;
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            // reads a line and breaks  it into tokens
            for(int j = 0; j < m; j++){
                int num = Integer.parseInt(st.nextToken());
                // takes next number and converts str to int
                if(num > max){
                    max = num;
                }
            }
        }
        System.out.println(max);
    }
}
